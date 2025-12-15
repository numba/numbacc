setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/sklam/sealir wip/updates_fixups deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/spylang/spy f53b7676bb36778a46156978a7a7694953bfed51 deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .


fmt:
	black -l79 ./nbcc


.PHONY: deps/spy deps/sealir
