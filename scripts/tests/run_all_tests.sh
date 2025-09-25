cd "$(dirname "$0")"

cd ..
sh ./compile.sh

cd tests

sh ./call_tests_cpu.sh
sh ./call_tests_gpu.sh
