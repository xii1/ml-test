# ML Test App

ML Test App is a REST API application which written by Python (using Flask).

## Getting Started

These instructions will get you building and running the project on your local machine for development and testing purposes. See usage and supported commands for notes on how to use the application.

## Prerequisites

- Python3+
- Docker

## Setup
```bash
./bin/setup
```

## Run for development
```bash
./bin/run
```

## Test sample APIs
```bash
http://localhost:8080/sample
http://localhost:8080/sample/hello
http://localhost:8080/visual/sample
http://localhost:8080/visual/water_potability
http://localhost:8080/classifier/dogcat
http://localhost:8080/classifier/handwritten
http://localhost:8080/classifier/fashion
http://localhost:8080/classifier/cifar10
http://localhost:8080/classifier/dogcat_heatmap
```

## Run for production
```bash
./bin/deploy_prod <tag> [scale]
```

## License
This project is licensed under the MIT License - see the LICENSE file for details
