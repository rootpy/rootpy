# simple makefile to simplify repetetive build env management tasks under posix

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-dict:
	rm -f AutoDict_*

clean-pyc:
	find rootpy -name "*.pyc" | xargs rm -f
	rm -f *.pyc

clean-so:
	find rootpy -name "*.so" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean-distribute:
	rm -f distribute-*.egg
	rm -f distribute-*.tar.gz

clean: clean-build clean-pyc clean-so clean-ctags clean-dict clean-distribute

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
	$(NOSETESTS) -a '!slow' -s rootpy

test-code-full: in
	$(NOSETESTS) -s rootpy

test-code-verbose: in
	$(NOSETESTS) -a '!slow' -s rootpy --nologcapture

test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture docs/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=rootpy rootpy

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
	rm -f __output.html

pep8:
	@pep8 --exclude=.git,extern rootpy

flakes:
	@./run-pyflakes
