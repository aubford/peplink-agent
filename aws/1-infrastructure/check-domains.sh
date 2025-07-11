#!/bin/bash

echo "Checking domain availability for langchain-pepwave project..."
echo "=================================================="

# List of potential domain names to check
domains=(
    "pepwave-ai.com"
    "langchain-pepwave.com"
    "pepwave-chat.com"
    "pepwave-assistant.com"
    "pepwave-help.com"
    "pepwave-ai.dev"
    "pepwave-ai.app"
    "langchain-pepwave.dev"
    "langchain-pepwave.app"
)

echo "Checking availability for these domains:"
echo

for domain in "${domains[@]}"; do
    echo -n "Checking $domain... "
    result=$(aws route53domains check-domain-availability --domain-name "$domain" --query 'Availability' --output text 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        if [ "$result" = "AVAILABLE" ]; then
            echo "✅ AVAILABLE"
        else
            echo "❌ $result"
        fi
    else
        echo "❓ Error checking (might not be supported)"
    fi
done

echo
echo "=================================================="
echo "To register a domain, choose one from the available list above and:"
echo "1. Update terraform.tfvars with your chosen domain"
echo "2. Uncomment the domain registration resource in main.tf"
echo "3. Fill in your contact information"
echo "4. Run 'terraform apply'"
echo
echo "Example terraform.tfvars entry:"
echo 'domain_name = "pepwave-ai.com"'
echo 'subdomain = "app"'
echo
echo "This will give you: https://app.pepwave-ai.com" 