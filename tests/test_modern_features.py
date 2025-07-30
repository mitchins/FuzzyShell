#!/usr/bin/env python3
"""Test modern Python 3.10+ features are working"""

import sys

def test_union_syntax():
    """Test that modern union syntax works"""
    def example_func(value: str | None = None) -> bool:
        return value is not None
    
    assert example_func("test") == True
    assert example_func(None) == False
    print("✅ Union syntax (str | None) works!")

def test_match_statement():
    """Test that match/case statements work"""
    def cycle_mode(mode: str) -> str:
        match mode:
            case "hybrid":
                return "semantic"
            case "semantic":
                return "keyword" 
            case "keyword":
                return "hybrid"
            case _:
                return "hybrid"  # fallback
    
    assert cycle_mode("hybrid") == "semantic"
    assert cycle_mode("semantic") == "keyword"
    assert cycle_mode("keyword") == "hybrid"
    assert cycle_mode("unknown") == "hybrid"
    print("✅ Match/case statements work!")

def test_generic_list_syntax():
    """Test that generic list syntax works"""
    def process_items(items: list[str]) -> int:
        return len(items)
    
    result = process_items(["a", "b", "c"])
    assert result == 3
    print("✅ Generic list[str] syntax works!")

if __name__ == "__main__":
    print(f"🐍 Testing modern Python features on Python {sys.version}")
    print("=" * 60)
    
    try:
        test_union_syntax()
        test_match_statement()
        test_generic_list_syntax()
        
        print("=" * 60)
        print("🎉 All modern Python 3.10+ features are working!")
        print("✨ Code is now more elegant and readable")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)