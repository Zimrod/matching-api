# railway_main.py
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
        """Find all buyer matches for a given listing"""
        try:
            print(f"🔍 DEBUG: Looking for listing ID: {listing_id}")
            
            listings = await self._get("listings", {"id": f"eq.{listing_id}"})
            
            print(f"🔍 DEBUG: Found {len(listings)} listings")
            for listing in listings:
                print(f"🔍 DEBUG: Listing ID: {listing.get('id')}")
            
            if not listings:
                logger.error(f"Listing {listing_id} not found")
                return []

            listing = listings[0]
            logger.info(f"Processing listing: {listing['product_data'].get('make')} {listing['product_data'].get('model')}")

            buyers = await self._get("buyers", {"select": "*"})
            matches = []

            for buyer in buyers:
                # Check if listing matches any of the buyer's preferences
                if self._is_match_with_preferences(listing, buyer):
                    match = self._create_match_record(listing, [buyer])
                    matches.append(match)

            if matches:
                await self._insert("matches", matches)
                logger.info(f"Created {len(matches)} matches for listing {listing_id}")

            return matches

        except Exception as e:
            logger.error(f"Error finding matches for listing {listing_id}: {str(e)}")
            return []

    def _is_match_with_preferences(self, listing: Dict[str, Any], buyer: Dict[str, Any]) -> bool:
        """Check if a listing matches any of the buyer's preferences"""
        try:
            preferences = buyer.get("preferences", [])
            
            # Handle both array and single object for backward compatibility
            if isinstance(preferences, dict):
                preferences = [preferences]
            elif not isinstance(preferences, list):
                preferences = []
            
            # If buyer has no preferences, no match
            if not preferences:
                return False
            
            # Check if listing matches ANY of the buyer's preferences
            for preference in preferences:
                if self._is_single_preference_match(listing, preference):
                    return True
            
            return False

        except Exception as e:
            logger.error(f"Error in match logic for buyer {buyer.get('name')}: {str(e)}")
            return False

    def _is_single_preference_match(self, listing: Dict[str, Any], preference: Dict[str, Any]) -> bool:
        """Check if listing matches a single preference object"""
        try:
            product_data = listing.get("product_data", {})
            
            # Extract listing details
            listing_make = str(product_data.get("make", "")).lower().strip()
            listing_model = str(product_data.get("model", "")).lower().strip()
            listing_price = self._safe_float(product_data.get("price", 0))
            listing_year = self._safe_int(product_data.get("year"))
            listing_category = str(product_data.get("category", "")).lower().strip()

            # Extract preference details
            pref_category = str(preference.get("category", "")).lower().strip()
            pref_vehicle = preference.get("vehicle", {})
            pref_price = preference.get("price", {})
            pref_location = preference.get("location", {})

            # 1. Check category match
            if pref_category and listing_category != pref_category:
                return False

            # 2. Check vehicle-specific matches (only if category is vehicles)
            if pref_category == "vehicles" and pref_vehicle:
                # Check make
                pref_make = str(pref_vehicle.get("make", "")).lower().strip()
                if pref_make and listing_make != pref_make:
                    return False

                # Check model
                pref_model = str(pref_vehicle.get("model", "")).lower().strip()
                if pref_model and listing_model != pref_model:
                    return False

                # Check year range
                pref_min_year = self._safe_int(pref_vehicle.get("minYear"))
                pref_max_year = self._safe_int(pref_vehicle.get("maxYear"))
                
                if listing_year:
                    if pref_min_year and listing_year < pref_min_year:
                        return False
                    if pref_max_year and listing_year > pref_max_year:
                        return False

            # 3. Check price range
            pref_min_price = self._safe_float(pref_price.get("min"))
            pref_max_price = self._safe_float(pref_price.get("max"))
            
            if listing_price > 0:  # Only check if listing has a valid price
                if pref_min_price and listing_price < pref_min_price:
                    return False
                if pref_max_price and listing_price > pref_max_price:
                    return False

            # 4. Check location (optional)
            pref_country = str(pref_location.get("country", "")).lower().strip()
            pref_city = str(pref_location.get("city", "")).lower().strip()
            
            listing_country = str(product_data.get("country", "")).lower().strip()
            listing_city = str(product_data.get("city", "")).lower().strip()
            
            if pref_country and listing_country and pref_country != listing_country:
                return False
            if pref_city and listing_city and pref_city != listing_city:
                return False

            logger.debug(f"Match found for preference: {pref_category} - {pref_vehicle.get('make', '')} {pref_vehicle.get('model', '')}")
            return True

        except Exception as e:
            logger.error(f"Error in single preference match: {str(e)}")
            return False

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

# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Message Processing Service...")
    print("✅ Matching service ready")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)