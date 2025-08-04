#!/bin/bash

# install pbjs / pbts
if [ ! -d node_modules ]; then
	npm install --save-dev protobufjs-cli
fi

# generate python code
protoc --proto_path=. --python_out=py ./questionbase.proto

# generate typescript code
mkdir -p js
node_modules/.bin/pbjs -t static-module -w es6 -o js/proto-bundle.js questionbase.proto
node_modules/.bin/pbjs -t static-module questionbase.proto | node_modules/.bin/pbts -o js/bundle.d.ts -

# fix import
sed -i 's/import \\* as \\$protobuf from/import \\$protobuf from/' js/proto-bundle.js
