#! /usr/bin/env bash

# This script assumes that the following repositories are checked out:

# https://github.com/akheron/cpython
# https://github.com/bbannier/ROOT
# https://github.com/pwaller/thisprefix

# It also assumes that it's running as sudo, although it will run everything
# except "make install" as the user you're sudoing from

set -e -u

# Place to install everything
DEST=/soft

# If we're inside sudo, run this as the normal user so as not to mess up
# permissions
function run-as-user() {
    if [[ -n "${SUDO_USER-}" ]]; then
        sudo -u "${SUDO_USER}" "$@"
        return
    fi
    "$@"
}

function build-python() {
    VERSION=${1}
    local INSTDIR=${DEST}/python/${VERSION}-dbg/
    
    if [[ -d "${INSTDIR}" ]]; then
        echo "Python ${VERSION} already available at ${INSTDIR}"
        echo "  (delete if if you want to rebuild)"
        return
    fi
    
    (cd cpython && git checkout ${VERSION})
    
    BDIR="cpython/build-${VERSION}-dbg"
    run-as-user mkdir -p $BDIR
    pushd $BDIR &> /dev/null
    
    run-as-user ../configure --enable-shared --with-pydebug --without-pymalloc --prefix=${INSTDIR}
    
    run-as-user make -j3
    
    make -j3 install
    
    if [[ -e "${DEST}/this.sh" ]]; then
        ln -s "${DEST}/this.sh" "${INSTDIR}/"
    fi
    
    sudo bash -c "source ${INSTDIR}/this.sh && python /soft/python/distribute-0.6.32/setup.py install && easy_install pip && pip install ipython nose virtualenv"
    
    popd &> /dev/null
}

function build-root() {
    local ROOT_VER="${1}"
    local ROOT_VER_NAME="${2}"
    local PYTHON_VER_NAME="${3}"
    
    local VERDIR="${ROOT_VER_NAME}-dbg-py${PYTHON_VER_NAME}"
    
    local INSTDIR=${DEST}/root/${VERDIR}
    
    if [[ -d "${INSTDIR}" ]]; then
        echo "ROOT ${ROOT_VER} already available at ${INSTDIR}"
        echo "  (delete if if you want to rebuild)"
        return
    fi
    
    (cd ROOT && git checkout ${ROOT_VER})
    
    BDIR="ROOT/build-${VERDIR}"
    run-as-user mkdir -p $BDIR
    pushd $BDIR &> /dev/null

        # --enable-fink
            
    ROOT_CONFIG="
        --build=debug
        --gminimal
        --with-cc=clang
        --with-cxx=clang++
        
        --prefix=${INSTDIR}
        --etcdir=${INSTDIR}/etc
        
        --enable-soversion
        --enable-rpath
        --enable-reflex
        --enable-xml
        
        --enable-python
        --with-python-libdir=/soft/python/${PYTHON_VER_NAME}-dbg/lib
        
        --enable-minuit2
    "
    run-as-user ../configure ${ROOT_CONFIG}
    
    run-as-user make -j3
    
    sudo make -j3 install
    
    sudo mkdir -p ${INSTDIR}
    
    sudo tee ${INSTDIR}/this.sh &> /dev/null <<EOF
source ${DEST}/python/${PYTHON_VER_NAME}-dbg/this.sh

function setup-root-cUF9F() {
    local THIS_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
    source \${THIS_DIR}/bin/thisroot.sh
    echo "\${BASH_SOURCE[0]}: using prefix"
}
setup-root-cUF9F
unset setup-root-cUF9F7

EOF
    
    popd &> /dev/null
}

# Note: this checks out a version according to ${*_VER} and installs it to ${*_VER_NAME}
function build-python-and-root() {
    PYTHON_VER="${1}"
    PYTHON_VER_NAME="${2}"
    ROOT_VER="${3}"
    ROOT_VER_NAME="${4}"
    shift 2
    build-python "$PYTHON_VER"
    
    source ${DEST}/python/${PYTHON_VER}-dbg/this.sh
    
    build-root "$ROOT_VER" "$ROOT_VER_NAME" "$PYTHON_VER_NAME"
    
    source ${DEST}/python/${PYTHON_VER}-dbg/this.sh remove
}

# Example invocation:
# build-python-and-root v2.7.3 2.7.3 v5-28-00 5.28
