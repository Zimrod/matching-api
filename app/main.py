# main.py
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import time
import httpx
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', 'your-supabase-url')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'your-supabase-key')

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation" 
}

logger = logging.getLogger(__name__)

class MatchingService:
    def __init__(self):
        self.base_url = f"{SUPABASE_URL}/rest/v1"

    async def _get(self, table: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = f"{SUPABASE_URL}/rest/v1/{table}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Accept": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()

    async def _insert(self, table: str, data: Any) -> List[Dict[str, Any]]:
        """Generic INSERT request"""
        async with httpx.AsyncClient() as client:
            try:
                headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                
                url = f"{self.base_url}/{table}"
                
                print("🔍 DEBUG _insert:")
                print(f"  URL: {url}")
                print(f"  Data: {data}")
                
                response = await client.post(
                    url,
                    headers=headers,
                    json=data,
                )
                
                print(f"🔍 DEBUG Response:")
                print(f"  Status: {response.status_code}")
                print(f"  Content: {response.text}")
                
                response.raise_for_status()
                result = response.json()
                print(f"✅ SUCCESS: {result}")
                return result
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                raise

    async def find_matches_for_listing(self, listing_id: str) -> List[Dict[str, Any]]:
        try:
            print(f"\n🔍 MATCHING: Fetching listing {listing_id}")

            listings = await self._get("listings", {"id": f"eq.{listing_id}"})
            if not listings:
                print(f"❌ No listing found for ID: {listing_id}")
                return []

            listing = listings[0]
            print(f"🚗 Processing: {listing['product_data'].get('make')} {listing['product_data'].get('model')}")

            buyers = await self._get("buyers", {"select": "*"})
            print(f"👥 Loaded {len(buyers)} buyers")

            matches = []

            for buyer in buyers:
                if self._matches_buyer(listing, buyer):
                    match = self._create_match_record(listing, buyer)
                    matches.append(match)

            # Insert individually
            for m in matches:
                await self._insert("matches", m)

            print(f"✅ TOTAL MATCHES CREATED: {len(matches)}")
            return matches

        except Exception as e:
            print(f"❌ ERROR in find_matches_for_listing: {e}")
            return []
        
    def _matches_buyer(self, listing: Dict[str, Any], buyer: Dict[str, Any]) -> bool:
        preferences = buyer.get("preferences", [])

        if isinstance(preferences, dict):
            preferences = [preferences]

        if not isinstance(preferences, list) or len(preferences) == 0:
            return False

        for pref in preferences:
            if self._match_single_preference(listing, pref, buyer):
                return True

        return False
    
    def _match_single_preference(self, listing, pref, buyer) -> bool:
        product = listing.get("product_data", {})

        # Normalize listing fields
        l_make = str(product.get("make", "")).lower()
        l_model = str(product.get("model", "")).lower()
        l_year = self._safe_int(product.get("year"))
        l_price = self._safe_float(product.get("price"))
        l_location_raw = (listing.get("location") or "").lower()

        # Normalize location from string: "Harare, Zimbabwe"
        l_city, l_country = self._parse_location(l_location_raw)

        # Normalize preference fields
        pref_vehicle = pref.get("vehicle", {})
        pref_price = pref.get("price", {})
        pref_location = pref.get("location", {})

        p_make = str(pref_vehicle.get("make", "")).lower()
        p_model = str(pref_vehicle.get("model", "")).lower()
        p_min_year = self._safe_int(pref_vehicle.get("minYear"))
        p_max_year = self._safe_int(pref_vehicle.get("maxYear"))
        p_min_price = self._safe_float(pref_price.get("min"))
        p_max_price = self._safe_float(pref_price.get("max"))

        p_city = str(pref_location.get("city", "")).lower().strip()
        p_country = str(pref_location.get("country", "")).lower().strip()

        print(f"\n🔎 Checking buyer {buyer.get('name')} against listing {product.get('make')} {product.get('model')}")

        # ---- 1. Make ----
        if p_make and l_make != p_make:
            print(f"❌ Make mismatch: Listing '{l_make}' != Preference '{p_make}'")
            return False

        # ---- 2. Model ----
        if p_model and l_model != p_model:
            print(f"❌ Model mismatch: Listing '{l_model}' != Preference '{p_model}'")
            return False

        # ---- 3. Year ----
        if l_year:
            if p_min_year and l_year < p_min_year:
                print(f"❌ Year too old: {l_year} < {p_min_year}")
                return False
            if p_max_year and l_year > p_max_year:
                print(f"❌ Year too new: {l_year} > {p_max_year}")
                return False

        # ---- 4. Price ----
        if l_price:
            if p_min_price and l_price < p_min_price:
                print(f"❌ Price too low: {l_price} < {p_min_price}")
                return False
            if p_max_price and l_price > p_max_price:
                print(f"❌ Price too high: {l_price} > {p_max_price}")
                return False

        # ---- 5. Location (OPTIONAL — only apply if both sides specify) ----
        if p_country:
            if not l_country:
                print(f"⚠️ Listing has no country but buyer requires '{p_country}' — ignoring")
            elif l_country != p_country:
                print(f"❌ Country mismatch: Listing '{l_country}' != Pref '{p_country}'")
                return False

        if p_city:
            if not l_city:
                print(f"⚠️ Listing has no city but buyer requires '{p_city}' — ignoring")
            elif l_city != p_city:
                print(f"❌ City mismatch: Listing '{l_city}' != Pref '{p_city}'")
                return False

        print(f"✅ MATCH SUCCESS for buyer {buyer.get('name')}")
        return True
    
    def _parse_location(self, location: str):
        if not location:
            return "", ""

        parts = [p.strip() for p in location.split(",")]

        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            # Could be either city OR country — but we treat as city
            return parts[0], ""
        return "", ""
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            if value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int"""
        try:
            if value is None:
                return None
            return int(value)
        except (ValueError, TypeError):
            return None

    def _create_match_record(self, listing: Dict[str, Any], buyers: Any) -> Dict[str, Any]:
        """Create a match record with one or multiple buyers"""
        if isinstance(buyers, dict):
            buyers = [buyers]
        
        buyer_data = []
        for buyer in buyers:
            buyer_data.append({
                "id": buyer["id"],
                "name": buyer["name"],
                "cell_number": buyer["cell_number"],
                "chat_id": buyer.get("chat_id")
            })
        
        return {
            "listing_id": listing["id"],
            "buyers": buyer_data,
            "product_data": listing["product_data"],
            "seller_id": listing.get("telegram_sender_id"),
            "seller_name": listing.get("seller_name", ""),
            "seller_contact": listing.get("seller_contact", ""),
            "matched_at": datetime.utcnow().isoformat(),
            "notified": False,
        }

# Create singleton instance
matching_service = MatchingService()

# Test function - moved inside the class
async def test_matching_locally():
    """Test the matching service with sample data"""
    # Fake listing
    listing = {
        "id": "test-listing-1",
        "category": "vehicles",
        "product_data": {
            "make": "Audi",
            "model": "A4",
            "year": 2023,
            "price": 36900,
            "currency": "USD"
        },
        "seller_name": "John Doe",
        "seller_contact": "+263771234567",
        "location": "Harare, Zimbabwe",
        "telegram_sender_id": 123456789
    }

    # Fake buyer
    buyer = {
        "id": "test-buyer-1",
        "name": "Alice",
        "cell_number": "+263772345678",
        "preferences": [
            {
                "category": "vehicles",
                "vehicle": {
                    "make": "Audi",
                    "model": "A4",
                    "minYear": 2018,
                    "maxYear": 2025
                },
                "price": {
                    "min": 30000,
                    "max": 50000
                },
                "location": {
                    "country": "Zimbabwe",
                    "city": "Harare"
                }
            }
        ]
    }

    # Test match
    result = matching_service._match_single_preference(listing, buyer["preferences"][0], buyer)
    print("Match result:", result)
    return result

# FastAPI App with lifespan (replaces on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Message Processing Service...")
    print("✅ Matching service ready")
    
    # Run test on startup
    print("🧪 Running local matching test...")
    await test_matching_locally()
    
    yield
    
    print("🛑 Shutting down Message Processing Service...")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Matching Service API"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/trigger-matching/{listing_id}")
async def trigger_matching(listing_id: str):
    matches = await matching_service.find_matches_for_listing(listing_id)
    return {
        "success": True,
        "listing_id": listing_id,
        "matches": matches,
        "match_count": len(matches),
    }

@app.post("/test-telegram-webhook")
async def test_telegram_webhook(payload: dict):
    try:
        message_data = {
            "raw_text": payload.get("raw_text", "Test message"),
            "sender_id": payload.get("sender_id", 123456789),
            "sender_username": payload.get("sender_username"),
            "sender_name": payload.get("sender_name", "Many Men"),
            "chat_id": payload.get("chat_id", -4846687198),
            "chat_title": payload.get("chat_title", "Test Group"),
            "message_id": payload.get("message_id", int(time.time())),
            "timestamp": payload.get("timestamp", time.time()),
        }
        
        N8N_FILTERING_URL = "https://primary-production-7bfa7.up.railway.app/webhook/telegram-messages"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                N8N_FILTERING_URL,
                json=message_data
            )
            
        return {
            "success": True,
            "sent_to_n8n": response.status_code == 200,
            "n8n_url": N8N_FILTERING_URL
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/test-local-matching")
async def test_local_matching_endpoint(payload: dict):
    """Endpoint to test matching with custom data"""
    listing = payload.get("listing")
    buyers = payload.get("buyers", [])

    matches = []
    for buyer in buyers:
        for pref in buyer.get("preferences", []):
            if matching_service._match_single_preference(listing, pref, buyer):
                matches.append({
                    "buyer_id": buyer["id"],
                    "buyer_name": buyer["name"]
                })

    return {"matches": matches}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)