# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "requests",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint

def main():
    print("Running UV PEP 723 Demo")
    
    # Simple Request
    try:
        response = requests.get("https://httpbin.org/get")
        data = response.json()
        
        print("\nUsing 'rich' to pretty print JSON response:")
        pprint(data)
        
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
