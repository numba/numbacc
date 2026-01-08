setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/sklam/sealir wip/updates_fixups deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/spylang/spy 53c2c65f91bdcf545bc751a497073fb649121988 deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .


fmt:
	black -l79 ./nbcc


.PHONY: deps/spy deps/sealir
