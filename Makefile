setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/numba/sealir 542a4e36a95926f34141cc2be9ed9e3e12d7aea7 deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/spylang/spy 1c04617d6916f875ddafffa9bd013a6508d28f79 deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .


fmt:
	black -l79 ./nbcc


test:
	pytest ./nbcc -v --benchmark-disable

.PHONY: deps/spy deps/sealir test fmt build
