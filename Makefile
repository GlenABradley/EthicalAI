# Simple helpers for OpenAPI and collections

BASE_URL ?= http://localhost:8080
DOCS_DIR ?= docs
OPENAPI_JSON := $(DOCS_DIR)/openapi.json
POSTMAN_JSON := $(DOCS_DIR)/postman_collection.json
THUNDER_JSON := $(DOCS_DIR)/thunder-collection_Coherence_API.json

.PHONY: openapi postman thunder collections all

openapi:
	python scripts/export_openapi.py --out $(OPENAPI_JSON)

postman: openapi
	python scripts/openapi_to_postman.py --in $(OPENAPI_JSON) --out $(POSTMAN_JSON) --base-url $(BASE_URL)

thunder: openapi
	python scripts/openapi_to_thunder.py --in $(OPENAPI_JSON) --out $(THUNDER_JSON) --base-url $(BASE_URL)

collections: openapi postman thunder

all: collections
	@echo "Artifacts written:"
	@echo "  $(OPENAPI_JSON)"
	@echo "  $(POSTMAN_JSON)"
	@echo "  $(THUNDER_JSON)"
