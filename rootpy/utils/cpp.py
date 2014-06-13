# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import re

from ..extern.pyparsing import (
    Optional, Keyword, Literal, Combine, Word, OneOrMore, QuotedString,
    delimitedList, ParseException, nums, alphas, alphanums, Group, Forward,
    Regex)
from .. import log; log = log[__name__]

__all__  = [
    'CPPGrammar',
]


class CPPGrammar(object):
    """
    A grammar for parsing C++ method/function signatures and types
    """
    ERROR_PATTERN = re.compile('\(line:\d+, col:(\d+)\)')

    QUOTED_STRING = (
        QuotedString('"', escChar='\\') | Literal('""') |
        QuotedString("'", escChar='\\') | Literal("''"))

    PTR = Combine(OneOrMore(Word("*") | Word("&")), adjacent=False)
    SIGN = Optional(Literal('+') | Literal('-'))
    CONST = Keyword("const")
    SIGNED = Optional(Keyword('signed') | Keyword('unsigned'))
    STATIC = Keyword('static')
    VOID = Combine(Literal('void') + Optional(PTR), adjacent=False)

    BASIC_TYPE = Group(
        Keyword('bool') |
        (SIGNED + Keyword('char')) |
        (SIGNED + Keyword('short')) |
        (SIGNED + Keyword('int')) |
        (SIGNED + Keyword('long') + Optional(Keyword('long'))) |
        Keyword('enum') |
        Keyword('float') |
        (Optional('long') + Keyword('double')))('type_name')

    IDENTIFIER = Word(alphas + "_", alphanums + "_").setName("identifier")

    hexnums = nums + "abcdefABCDEF" + "_?"
    base = Regex("'[bBoOdDhH]").setName("base")
    BASEDNUMBER = Combine(
        Optional(Word(nums + "_")) + base +
        Word(hexnums + "xXzZ"),
        joinString=" ", adjacent=False).setName("based number")
    NUMBER = (
        BASEDNUMBER |
        Regex(r"[-+]?([0-9]*\.[0-9]+|[0-9]+\.?)([Ee][-+]?[0-9]+)?")
        ).setName("numeric")

    ARITH_OPERATOR = Word("*/+-").setName('arith op')
    BIT_OPERATOR = Word('&|').setName('bit op')
    BIT_EXPRESSION = (IDENTIFIER +
        OneOrMore(BIT_OPERATOR + IDENTIFIER)).setName('bit_expression')
    EXPRESSION = OneOrMore(
        NUMBER | ARITH_OPERATOR | IDENTIFIER).setName('expression')
    FULL_EXPRESSION = OneOrMore(
        NUMBER | ARITH_OPERATOR |
        Literal('(') | Literal(')')).setName('full_expression')

    NAMESPACED_NAME = (Optional(Literal('::')).suppress() +
            delimitedList(IDENTIFIER, delim='::', combine=True))

    TYPE = Forward()

    TEMPLATE_PARAMS = (
        Literal("<").suppress() +
        Group(delimitedList(TYPE | FULL_EXPRESSION))("template_params") +
        Literal(">").suppress())

    CLASS_MEMBER = (
        Literal('::').suppress() +
        NAMESPACED_NAME)("template_member")

    COMPLEX_TYPE = (
        Group(NAMESPACED_NAME)('type_name') +
        Optional(TEMPLATE_PARAMS + Optional(CLASS_MEMBER)))

    TYPE << (
        Group(Optional(CONST))('type_prefix') +
        (BASIC_TYPE | COMPLEX_TYPE) +
        Group(Optional(PTR + Optional(CONST)))('type_suffix'))

    TYPE_EXPRESSION = (
        Optional(NUMBER + ARITH_OPERATOR) + TYPE)('type_expression')

    SIMPLE_ARG_DEFAULT = Forward()
    ARG_LIST = delimitedList(SIMPLE_ARG_DEFAULT)('arg_list')
    SIMPLE_ARG_DEFAULT << (
        QUOTED_STRING | BIT_EXPRESSION | (TYPE_EXPRESSION +
        Optional(
            Literal('(').suppress() +
            Optional(ARG_LIST) +
            Literal(')').suppress())) |
        EXPRESSION)

    FUNC_ARG_DEFAULT = (Literal('&') + TYPE_EXPRESSION +
        Optional(Literal('(').suppress() +
        Optional(ARG_LIST) +
        Literal(')').suppress()))

    METHOD_ARGS = Forward()

    SIMPLE_ARG = (Optional(Optional(IDENTIFIER('arg_name')) +
        Optional(Literal('=').suppress() +
            SIMPLE_ARG_DEFAULT('arg_default'))))

    FUNC_ARG = (Literal('(*)') +
        Literal("(").suppress() +
        Optional(METHOD_ARGS)('func_arg_signature') +
        Literal(")").suppress() +
        Optional(Optional(IDENTIFIER)('arg_name') +
            Optional(Literal('=').suppress() +
                FUNC_ARG_DEFAULT('arg_default'))))

    METHOD_ARG = Group((VOID('arg_type') | TYPE('arg_type')) +
        (FUNC_ARG | SIMPLE_ARG))('arg')

    METHOD_ARGS << (delimitedList(METHOD_ARG) | Literal('...'))

    METHOD_SIGNATURE = (
        Optional(STATIC) +
        ((VOID('return') + COMPLEX_TYPE('name')) |
         (TYPE('return') + COMPLEX_TYPE('name')) |
         COMPLEX_TYPE('name')) +
        Literal("(").suppress() +
        Optional(Group(METHOD_ARGS)('args')) +
        Literal(")").suppress())

    @classmethod
    def _parse(cls, grammar, string, raise_exception=False, silent=True):
        try:
            return grammar.parseString(string, parseAll=True)
        except ParseException as e:
            if not silent:
                log.warning(string)
                str_e = str(e)
                match = re.search(cls.ERROR_PATTERN, str_e)
                if match:
                    log.warning(" " * (int(match.group(1)) - 1) + '^')
            if raise_exception:
                raise
            if not silent:
                log.warning(e)
            return None

    @classmethod
    def parse_type(cls, string, raise_exception=False, silent=True):
        return cls._parse(cls.TYPE, string,
                          raise_exception, silent)

    @classmethod
    def parse_method(cls, string, raise_exception=False, silent=True):
        return cls._parse(cls.METHOD_SIGNATURE,
                          string, raise_exception, silent)
