# Looking to build the library alone?
# Just dump fast_oscar.h in your project and make sure to compile with -std=c++11
# This file builds the tests and runs them

set -e
set -u

CXX_FLAGS="-g -Wall -Wextra -Werror -std=c++11"
GTEST_CXX_FLAGS="-pthread"
GTEST_DIR=gtest-1.7.0
GTEST_INC=$GTEST_DIR/include
GTEST_LIB=$GTEST_DIR/lib

# Grab the gtest dependency
if [[ ! -e $GTEST_DIR ]]; then
    wget 'http://googletest.googlecode.com/files/gtest-1.7.0.zip' -O gtest-1.7.0.zip
    unzip gtest-1.7.0.zip
    (
        pushd $GTEST_DIR
        ./configure
        make -j8
        popd
    )
fi

# Build the tests
set -x
g++ $CXX_FLAGS $GTEST_CXX_FLAGS -I$GTEST_INC $GTEST_LIB/.libs/libgtest_main.a $GTEST_LIB/.libs/libgtest.a fast_oscar_test.cpp -o fast_oscar_test

# Run the tests
./fast_oscar_test
