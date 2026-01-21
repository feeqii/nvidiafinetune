#!/usr/bin/env python3
"""Fetch Quranic text with diacritics from quran.com API."""

import json
import requests

def fetch_surah_with_diacritics(surah_number):
    """Fetch surah text with diacritics from quran.com API.
    
    Args:
        surah_number: Surah number (1 for Fatiha, 112 for Ikhlas)
    
    Returns:
        Dict with surah info and verses
    """
    # Try the quran.com API v4
    url = f"https://api.quran.com/api/v4/verses/by_chapter/{surah_number}"
    params = {
        "language": "ar",
        "words": "false",
        "translations": "",
        "audio": "false",
        "tafsirs": ""
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        verses = []
        full_text_parts = []
        
        for verse in data.get("verses", []):
            # text_uthmani has full diacritics
            verse_text = verse.get("text_uthmani", "")
            verses.append(verse_text)
            full_text_parts.append(verse_text)
        
        return {
            "verses": verses,
            "full_text": " ".join(full_text_parts),
            "verse_count": len(verses)
        }
    
    except Exception as e:
        print(f"Error fetching surah {surah_number}: {e}")
        return None

def main():
    print("Fetching Al-Fatiha (Surah 1)...")
    fatiha = fetch_surah_with_diacritics(1)
    
    print("Fetching Al-Ikhlas (Surah 112)...")
    ikhlas = fetch_surah_with_diacritics(112)
    
    if fatiha and ikhlas:
        result = {
            "fatiha": fatiha,
            "ikhlas": ikhlas
        }
        
        # Save to file
        output_file = "quranic_texts_with_diacritics.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Successfully saved to {output_file}")
        print(f"\nFatiha preview: {fatiha['full_text'][:100]}...")
        print(f"Ikhlas preview: {ikhlas['full_text'][:100]}...")
    else:
        print("Failed to fetch texts")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
