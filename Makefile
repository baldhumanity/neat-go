.PHONY: test lint fmt clean run-xor

# Default target
all: fmt lint test

# Run tests
test:
	go test ./...

# Run linter
lint:
	go vet ./...
	@if command -v golangci-lint > /dev/null; then \
		echo "Running golangci-lint"; \
		golangci-lint run; \
	else \
		echo "golangci-lint not found, skipping"; \
	fi

# Format the code
fmt:
	go fmt ./...

# Clean build artifacts
clean:
	go clean
	rm -f coverage.out

# Run the XOR example
run-xor:
	cd examples/xor && go run main.go

# Generate documentation
docs:
	godoc -http=:6060

# Build the project
build:
	go build ./...

# Install dependencies
deps:
	go get -v -t -d ./...

# Generate coverage report
cover:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out 