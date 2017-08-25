# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)
NOSETESTS := $(shell which nosetests)

INTERACTIVE := $(shell ([ -t 0 ] && echo 1) || echo 0)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
        OPEN := open
else
        OPEN := xdg-open
endif

STANDARD_TEST_ATTR="not slow and not network"

all: clean inplace test

# list what would be deleted by clean-repo
clean-repo-check:
	@git clean -f -x -d -n

# remove all untracked files and directories
clean-repo:
	@git clean -f -x -d

clean-dict:
	@rm -f AutoDict_*

clean-pyc:
	@find . -name "*.pyc" -exec rm {} \;

clean-so:
	@find rootpy -name "*.so" -exec rm {} \;

clean-build:
	@rm -rf build

clean-examples:
	@find examples -name "*.root" -exec rm {} \;
	@find examples -name "*.h5" -exec rm {} \;
	@find examples -name "*.gif" -exec rm {} \;

clean: clean-build clean-pyc clean-so clean-dict clean-examples

in: inplace # just a shortcut
inplace:
	@$(PYTHON) setup.py build_ext -i

install: clean
	@$(PYTHON) setup.py install

install-user: clean
	@$(PYTHON) setup.py install --user

sdist: clean
	@$(PYTHON) setup.py sdist --release

test-code: inplace
	@$(NOSETESTS) -v -A $(STANDARD_TEST_ATTR) -s --exclude=extern rootpy

test-code-full: inplace
	@$(NOSETESTS) -v -s --exclude=extern rootpy

test-code-verbose: inplace
	@$(NOSETESTS) -v -A $(STANDARD_TEST_ATTR) -s --exclude=extern rootpy --nologcapture

test-installed:
	@(mkdir -p nose && cd nose && \
	$(NOSETESTS) -v -A $(STANDARD_TEST_ATTR) -s --exe rootpy && \
	cd .. && rm -rf nose)

test-doc:
	@$(NOSETESTS) -v -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture docs/

test-coverage:
	@rm -rf coverage .coverage
	@$(NOSETESTS) -s -v -A $(STANDARD_TEST_ATTR) --with-coverage \
		--cover-erase --cover-branches \
		--cover-html --cover-html-dir=coverage \
		--exclude=extern rootpy
	@if [ "$(INTERACTIVE)" -eq "1" ]; then \
		$(OPEN) coverage/index.html; \
	fi;

test-examples: clean-examples
	@PYTHONPATH=$(PWD):$(PYTHONPATH); \
	for example in `find examples -name "*.py"`; do \
	    echo; \
	    echo Running $$example ...; \
	    if !(cd `dirname $$example` && ROOTPY_BATCH=1 $(PYTHON) `basename $$example`); then \
	    	echo $$example failed!; \
	    	exit 1; \
	    fi; \
	done

test: test-code test-doc

trailing-spaces:
	@find rootpy -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

doc-clean:
	@make -C docs/ clean

doc: doc-clean inplace
	@make -C docs/ html

check-rst:
	@mkdir -p build
	@$(PYTHON) setup.py --long-description | rst2html.py > build/README.html
	@$(OPEN) build/README.html

pep8:
	@pep8 --exclude=.git,extern rootpy

gh-pages: doc
	@./ghp-import -m "update docs" -r upstream -f -p docs/_build/html/
