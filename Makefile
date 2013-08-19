# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)
NOSETESTS ?= nosetests

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

clean-distribute:
	@rm -f distribute-*.egg
	@rm -f distribute-*.tar.gz

clean-examples:
	@find examples -name "*.root" -exec rm {} \;
	@find examples -name "*.h5" -exec rm {} \;

clean: clean-build clean-pyc clean-so clean-dict clean-distribute clean-examples

in: inplace # just a shortcut
inplace:
	@$(PYTHON) setup.py build_ext -i

install: clean
	@$(PYTHON) setup.py install

install-user: clean
	@$(PYTHON) setup.py install --user

sdist: clean
	@$(PYTHON) setup.py sdist --release

register:
	@$(PYTHON) setup.py register --release

upload: clean
	@$(PYTHON) setup.py sdist upload --release

test-code: inplace
	@$(NOSETESTS) -v -a '!slow' -s rootpy

test-code-full: inplace
	@$(NOSETESTS) -v -s rootpy

test-code-verbose: inplace
	@$(NOSETESTS) -v -a '!slow' -s rootpy --nologcapture

test-installed:
	@(mkdir -p nose && cd nose && \
	$(NOSETESTS) -v -a '!slow' -s --exe rootpy && \
	cd .. && rm -rf nose)

test-doc:
	@$(NOSETESTS) -v -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture docs/

test-coverage:
	@rm -rf coverage .coverage
	@$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=rootpy rootpy

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

doc: inplace
	@make -C docs/ html

update-distribute:
	@curl -O http://python-distribute.org/distribute_setup.py

check-rst:
	@python setup.py --long-description | rst2html.py > __output.html
	@firefox __output.html
	@rm -f __output.html

pep8:
	@pep8 --exclude=.git,extern rootpy

flakes:
	@./run-pyflakes

gh-pages:
	@./ghp-import -m "update docs" -r upstream -f -p docs/_build/html/

upload-docs:
	@cd docs/_build/html && scp -r * rootpyor@rootpy.org:~/public_html/
