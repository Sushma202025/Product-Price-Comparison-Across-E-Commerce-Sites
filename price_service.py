from statistics import median
import requests
import os
import re

SERP_API_KEY = os.getenv("SERP_API_KEY")

def clean_price(price):
    if not price:
        return None
    return int(re.sub(r"[^\d]", "", price))


def search_products(query):

    params = {
        "engine": "google_shopping",
        "q": query,
        "hl": "en",
        "gl": "in",
        "api_key": SERP_API_KEY
    }

    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()

    results = []

    # 1️⃣ Take first 5 results
    for item in data.get("shopping_results", []):
        store = item.get("source")
        price = item.get("price")
        link = item.get("product_link") or item.get("link")

        if not store or not price or not link:
            continue

        num_price = clean_price(price)
        if num_price is None:
            continue

        results.append({
            "store": store,
            "price": f"₹{num_price:,}",
            "price_value": num_price,
            "link": link
        })

        if len(results) == 5:
            break

    # If 0 or 1 item → return directly
    if len(results) <= 1:
        return results, results[0] if results else None

    # 2️⃣ Sort and remove lowest price (always wrong)
    results = sorted(results, key=lambda x: x["price_value"])
    removed_lowest = results.pop(0)

    # now 4 remain
    if len(results) <= 2:
        return results, results[0]

    # 3️⃣ Cluster remaining by ±15%
    values = [x["price_value"] for x in results]
    mid = median(values)
    low = mid * 0.85
    high = mid * 1.15

    clustered = [r for r in results if low <= r["price_value"] <= high]

    # 4️⃣ If cluster removes everything → return remaining 4
    if not clustered:
        return results, results[0]

    clustered = sorted(clustered, key=lambda x: x["price_value"])

    return clustered, clustered[0]
