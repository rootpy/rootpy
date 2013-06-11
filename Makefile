# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-dict:
	rm -f AutoDict_*

clean-pyc:
	find . -name "*.pyc" -exec rm {} \;

clean-so:
	find rootpy -name "*.so" -exec rm {} \;

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean-distribute:
	rm -f distribute-*.egg
	rm -f distribute-*.tar.gz

clean-examples:
	find examples -name "*.root" -exec rm {} \;
	find examples -name "*.h5" -exec rm {} \;

clean: clean-build clean-pyc clean-so \
	clean-ctags clean-dict clean-distribute clean-examples

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

install:
	$(PYTHON) setup.py install

install-user:
	$(PYTHON) setup.py install --user

sdist: clean
	$(PYTHON) setup.py sdist --release

register:
	$(PYTHON) setup.py register --release

upload: clean
	$(PYTHON) setup.py sdist upload --release

test-code: in
	$(NOSETESTS) -v -a '!slow' -s rootpy

test-code-full: in
	$(NOSETESTS) -v -s rootpy

test-code-verbose: in
	$(NOSETESTS) -v -a '!slow' -s rootpy --nologcapture

test-doc:
	$(NOSETESTS) -v -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture docs/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=rootpy rootpy

test-examples: clean-examples
	@PYTHONPATH=$(PWD):$(PYTHONPATH); \
	for example in `find examples -name "*.py"`; do \
	    echo; \
	    echo Running $$example ...; \
	    (cd `dirname $$example` && ROOTPY_BATCH=1 $(PYTHON) `basename $$example`) \
	done

test: test-code test-doc

trailing-spaces:
	find rootpy -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

doc: inplace docs/_themes/sphinx-bootstrap/bootstrap.js
	make -C docs/ html

docs/_themes/sphinx-bootstrap/bootstrap.js:
	echo "Did not find docs/_themes/sphinx-bootstrap, which is needed to make the docs."
	echo "Downloading it now for you as a git submodule."
	echo "This will fail if you don't have internet connection."
	git submodule init
	git submodule update

update-distribute:
	curl -O http://python-distribute.org/distribute_setup.py

check-rst:
	python setup.py --long-description | rst2html.py > __output.html
	firefox __output.html
	rm -f __output.html

pep8:
	@pep8 --exclude=.git,extern rootpy

flakes:
	@./run-pyflakes

gh-pages:
	@./ghp-import -m "update docs" -r upstream -f -p docs/_build/html/

upload-docs:
	@cd docs/_build/html && scp -r * rootpyor@rootpy.org:~/public_html/
