rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release -- -j 32