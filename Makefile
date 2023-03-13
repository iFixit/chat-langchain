.PHONY: start
start:
	uvicorn main:app --reload --port 9100

.PHONY: format
format:
	black .
	isort .