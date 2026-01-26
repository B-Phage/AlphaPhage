#!/bin/bash
# tests/ops/smoke_test.sh

echo "Checking if service is up..."
# Replace with your actual health check endpoint
# This could be a stremlit ui that serves inference for example
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://your-production-url.com/health)

if [ $STATUS -eq 200 ]; then
  echo "Deployment Successful!"
  exit 0
else
  echo "Deployment Failed with status $STATUS"
  exit 1
fi
