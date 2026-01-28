setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/numba/sealir 69972b22636fe8d8c22cc8b745a6f2fa8e763733 deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/spylang/spy cfeae1306f91694b7f672a190e478952e9e4257f deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .


fmt:
	black -l79 ./nbcc


test:
	pytest ./nbcc/tests -v --benchmark-disable


test-cuda-tile:
	pytest ./nbcc/cutile_backend/tests -v --benchmark-disable

.PHONY: deps/spy deps/sealir test fmt build
