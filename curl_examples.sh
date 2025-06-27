#!/bin/bash
# LLM Platform API Examples using curl

BASE_URL="http://localhost:5000/api"

echo "=== LLM Platform API - curl Examples ==="
echo

# 1. Register a user and get API key
echo "1. Register User and Get API Key"
echo "================================"
echo "curl -X POST $BASE_URL/auth/register \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"username\": \"curl_user\","
echo "    \"email\": \"curl@example.com\","
echo "    \"password\": \"password123\""
echo "  }'"
echo

# Save the API key (you'll need to replace this with actual key from response)
API_KEY="llm_demo_key_12345678901234567890"
echo "# Replace API_KEY with your actual key from registration response"
echo "API_KEY=\"$API_KEY\""
echo

# 2. List models (no auth required)
echo "2. List Available Models"
echo "========================"
echo "curl -X GET $BASE_URL/models"
echo

# 3. Generate text (requires auth)
echo "3. Generate Text (Authenticated)"
echo "================================"
echo "curl -X POST $BASE_URL/models/1/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'X-API-Key: \$API_KEY' \\"
echo "  -d '{"
echo "    \"prompt\": \"def fibonacci(n):\","
echo "    \"temperature\": 0.7,"
echo "    \"max_length\": 100"
echo "  }'"
echo

# 4. List coding datasets
echo "4. List Coding Datasets"
echo "======================="
echo "curl -X GET $BASE_URL/datasets"
echo

# 5. Create coding training job (requires auth)
echo "5. Create Coding Training Job (Authenticated)"
echo "=============================================="
echo "curl -X POST $BASE_URL/training/coding \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'X-API-Key: \$API_KEY' \\"
echo "  -d '{"
echo "    \"model_id\": 1,"
echo "    \"job_name\": \"Python Code Training\","
echo "    \"dataset_id\": 1,"
echo "    \"epochs\": 5,"
echo "    \"learning_rate\": 0.00005,"
echo "    \"batch_size\": 4"
echo "  }'"
echo

# 6. Check training job status
echo "6. Check Training Job Status"
echo "============================"
echo "curl -X GET $BASE_URL/training/1"
echo

# 7. List your API keys (requires auth)
echo "7. List Your API Keys (Authenticated)"
echo "====================================="
echo "curl -X GET $BASE_URL/auth/keys \\"
echo "  -H 'X-API-Key: \$API_KEY'"
echo

# 8. Create new API key (requires auth)
echo "8. Create New API Key (Authenticated)"
echo "====================================="
echo "curl -X POST $BASE_URL/auth/keys \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'X-API-Key: \$API_KEY' \\"
echo "  -d '{"
echo "    \"key_name\": \"My Second Key\","
echo "    \"rate_limit\": 500"
echo "  }'"
echo

# 9. Run model evaluation (requires auth)
echo "9. Run Model Evaluation (Authenticated)"
echo "======================================="
echo "curl -X POST $BASE_URL/models/1/evaluate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'X-API-Key: \$API_KEY' \\"
echo "  -d '{"
echo "    \"eval_name\": \"Custom Evaluation\""
echo "  }'"
echo

# 10. Get platform statistics
echo "10. Get Platform Statistics"
echo "==========================="
echo "curl -X GET $BASE_URL/statistics"
echo

# 11. Error examples
echo "11. Error Handling Examples"
echo "==========================="
echo "# Missing API key (401 error):"
echo "curl -X POST $BASE_URL/models/1/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"prompt\": \"test\"}'"
echo
echo "# Invalid API key (401 error):"
echo "curl -X POST $BASE_URL/models/1/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'X-API-Key: invalid_key' \\"
echo "  -d '{\"prompt\": \"test\"}'"
echo

echo "=== Authentication & Rate Limiting Notes ==="
echo "• Include X-API-Key header for authenticated endpoints"
echo "• Default rate limit: 1000 requests/day per API key"
echo "• Rate limits reset daily at midnight UTC"
echo "• Monitor usage via GET /auth/keys endpoint"
echo "• HTTP 401 = Authentication error"
echo "• HTTP 429 = Rate limit exceeded"
echo

echo "=== Quick Start ==="
echo "1. Register: curl -X POST $BASE_URL/auth/register -H 'Content-Type: application/json' -d '{\"username\":\"user\",\"email\":\"user@example.com\",\"password\":\"pass\"}'"
echo "2. Save API key from response"
echo "3. Use API key in X-API-Key header for authenticated endpoints"
echo "4. Check rate limit usage periodically"