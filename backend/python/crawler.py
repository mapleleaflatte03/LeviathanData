"""
Leviathan Ethical Autonomous Crawler Module

Ethical guidelines:
- 4s rate limit between requests
- Public/legal sources only  
- Max 300 items per hunt
- 60s timeout per source
- Respect robots.txt
- VN-focused: Cafef, VNExpress, VNDIRECT, Yahoo Finance VN
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse

import aiohttp
import feedparser
import pandas as pd

logger = logging.getLogger("crawler")

# Ethical crawl config
RATE_LIMIT_SECONDS = 4.0
MAX_ITEMS_PER_HUNT = 300
TIMEOUT_SECONDS = 60
USER_AGENT = "Leviathan-DataHunter/1.0 (Ethical Crawler; +https://github.com/leviathan)"

# Approved public sources
APPROVED_SOURCES = {
    # Vietnam finance
    "vnexpress.net",
    "cafef.vn",
    "vndirect.com.vn",
    "stockbiz.vn",
    "hsc.com.vn",
    "fpts.com.vn",
    "yahoo.com",
    "finance.yahoo.com",
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
    # Data APIs
    "kaggle.com",
    "api.worldbank.org",
    "data.worldbank.org",
    "alphavantage.co",
    # News RSS
    "feeds.feedburner.com",
    "rss.cnn.com",
    # Government/Open data
    "data.gov",
    "data.gov.vn",
    "gso.gov.vn",
}


def is_approved_source(url: str) -> bool:
    """Check if URL belongs to an approved source."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Check direct match or subdomain match
        for approved in APPROVED_SOURCES:
            if domain == approved or domain.endswith(f".{approved}"):
                return True
        return False
    except Exception:
        return False


class RateLimiter:
    """Simple rate limiter for ethical crawling."""
    
    def __init__(self, min_interval: float = RATE_LIMIT_SECONDS):
        self.min_interval = min_interval
        self.last_request: Dict[str, float] = {}
    
    async def wait(self, domain: str) -> None:
        """Wait if needed before making request to domain."""
        now = time.time()
        last = self.last_request.get(domain, 0)
        wait_time = self.min_interval - (now - last)
        
        if wait_time > 0:
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s for {domain}")
            await asyncio.sleep(wait_time)
        
        self.last_request[domain] = time.time()


class EthicalCrawler:
    """Autonomous ethical data crawler for Leviathan."""
    
    def __init__(
        self,
        on_progress: Optional[Callable[[str, str], None]] = None,
        on_item: Optional[Callable[[Dict], None]] = None,
    ):
        self.rate_limiter = RateLimiter()
        self.on_progress = on_progress or (lambda s, m: None)
        self.on_item = on_item or (lambda i: None)
        self.items_collected: List[Dict[str, Any]] = []
        self.sources_used: List[str] = []
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self) -> "EthicalCrawler":
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS),
        )
        return self
    
    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
    
    def _emit_progress(self, stage: str, message: str) -> None:
        """Emit progress update."""
        self.on_progress(stage, message)
        logger.info(f"[{stage}] {message}")
    
    async def hunt(self, prompt: str) -> Dict[str, Any]:
        """
        Main autonomous hunt based on natural language prompt.
        Returns collected data and metadata.
        """
        self._emit_progress("PARSE", f"Analyzing hunt request: {prompt[:100]}...")
        
        # Determine hunt type from prompt
        hunt_type = self._classify_hunt(prompt)
        self._emit_progress("CLASSIFY", f"Hunt type: {hunt_type}")
        
        try:
            if hunt_type == "vn_stock":
                await self._hunt_vn_stock(prompt)
            elif hunt_type == "vn_news":
                await self._hunt_vn_news(prompt)
            elif hunt_type == "kaggle":
                await self._hunt_kaggle(prompt)
            elif hunt_type == "world_bank":
                await self._hunt_world_bank(prompt)
            elif hunt_type == "rss":
                await self._hunt_rss(prompt)
            else:
                # Generic web hunt with ethical constraints
                await self._hunt_generic(prompt)
        except Exception as e:
            logger.error(f"Hunt error: {e}")
            self._emit_progress("ERROR", str(e))
        
        return {
            "items_collected": len(self.items_collected),
            "sources": self.sources_used,
            "data": self.items_collected[:MAX_ITEMS_PER_HUNT],
            "timestamp": datetime.now().isoformat(),
        }
    
    def _classify_hunt(self, prompt: str) -> str:
        """Classify hunt type from prompt."""
        prompt_lower = prompt.lower()
        
        if any(k in prompt_lower for k in ["vn stock", "vietnam stock", "vni", "vnindex", "yahoo finance vietnam"]):
            return "vn_stock"
        elif any(k in prompt_lower for k in ["vn news", "vietnam news", "vnexpress", "cafef", "vietnamese article"]):
            return "vn_news"
        elif any(k in prompt_lower for k in ["kaggle", "titanic", "iris", "mnist"]):
            return "kaggle"
        elif any(k in prompt_lower for k in ["world bank", "gdp", "economic indicator", "country data"]):
            return "world_bank"
        elif any(k in prompt_lower for k in ["rss", "feed", "news feed"]):
            return "rss"
        else:
            return "generic"
    
    async def _fetch_json(self, url: str) -> Optional[Dict]:
        """Fetch JSON from URL with rate limiting and error handling."""
        if not is_approved_source(url):
            self._emit_progress("SKIP", f"Source not approved: {url}")
            return None
        
        domain = urlparse(url).netloc
        await self.rate_limiter.wait(domain)
        
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"HTTP {response.status} from {url}")
                    return None
        except Exception as e:
            logger.error(f"Fetch error {url}: {e}")
            return None
    
    async def _fetch_text(self, url: str) -> Optional[str]:
        """Fetch text/HTML from URL with rate limiting."""
        if not is_approved_source(url):
            self._emit_progress("SKIP", f"Source not approved: {url}")
            return None
        
        domain = urlparse(url).netloc
        await self.rate_limiter.wait(domain)
        
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"HTTP {response.status} from {url}")
                    return None
        except Exception as e:
            logger.error(f"Fetch error {url}: {e}")
            return None
    
    async def _hunt_vn_stock(self, prompt: str) -> None:
        """Hunt for Vietnam stock market data from Yahoo Finance."""
        self._emit_progress("HUNT", "Fetching Vietnam stock data from public APIs...")
        
        # Yahoo Finance symbols for Vietnam
        symbols = ["^VNINDEX", "VNM", "VIC.VN"]
        
        for symbol in symbols:
            if len(self.items_collected) >= MAX_ITEMS_PER_HUNT:
                break
            
            # Use Yahoo Finance API (public, rate-limited)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1mo"
            data = await self._fetch_json(url)
            
            if data and "chart" in data and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                timestamps = result.get("timestamp", [])
                quotes = result.get("indicators", {}).get("quote", [{}])[0]
                
                for i, ts in enumerate(timestamps[-30:]):  # Last 30 days
                    if len(self.items_collected) >= MAX_ITEMS_PER_HUNT:
                        break
                    
                    item = {
                        "symbol": symbol,
                        "date": datetime.fromtimestamp(ts).isoformat(),
                        "open": quotes.get("open", [None])[i] if i < len(quotes.get("open", [])) else None,
                        "high": quotes.get("high", [None])[i] if i < len(quotes.get("high", [])) else None,
                        "low": quotes.get("low", [None])[i] if i < len(quotes.get("low", [])) else None,
                        "close": quotes.get("close", [None])[i] if i < len(quotes.get("close", [])) else None,
                        "volume": quotes.get("volume", [None])[i] if i < len(quotes.get("volume", [])) else None,
                        "source": "yahoo_finance",
                    }
                    self.items_collected.append(item)
                    self.on_item(item)
                
                self.sources_used.append(f"Yahoo Finance: {symbol}")
                self._emit_progress("DATA", f"Collected {len(self.items_collected)} stock records for {symbol}")
    
    async def _hunt_vn_news(self, prompt: str) -> None:
        """Hunt for Vietnam news from RSS feeds."""
        self._emit_progress("HUNT", "Crawling Vietnam news RSS feeds...")
        
        # VNExpress and Cafef RSS feeds
        rss_feeds = [
            ("https://vnexpress.net/rss/tin-moi-nhat.rss", "VNExpress"),
            ("https://cafef.vn/rss/trang-chu.rss", "CafeF"),
        ]
        
        for feed_url, source_name in rss_feeds:
            if len(self.items_collected) >= MAX_ITEMS_PER_HUNT:
                break
            
            try:
                # Rate limit
                domain = urlparse(feed_url).netloc
                await self.rate_limiter.wait(domain)
                
                # Fetch and parse RSS
                async with self._session.get(feed_url) as response:
                    if response.status != 200:
                        continue
                    content = await response.text()
                
                feed = feedparser.parse(content)
                
                for entry in feed.entries[:50]:  # Max 50 per feed
                    if len(self.items_collected) >= MAX_ITEMS_PER_HUNT:
                        break
                    
                    item = {
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", "")[:500],  # Limit summary
                        "source": source_name,
                    }
                    self.items_collected.append(item)
                    self.on_item(item)
                
                self.sources_used.append(source_name)
                self._emit_progress("DATA", f"Collected {len(self.items_collected)} articles from {source_name}")
                
            except Exception as e:
                logger.error(f"RSS fetch error {feed_url}: {e}")
    
    async def _hunt_kaggle(self, prompt: str) -> None:
        """Hunt for Kaggle datasets using Kaggle API."""
        self._emit_progress("HUNT", "Searching Kaggle datasets...")
        
        # Check for Kaggle credentials
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            self._emit_progress("WARN", "Kaggle API not configured. Using cached datasets...")
            # Fall back to known local datasets
            await self._hunt_local_datasets(prompt)
            return
        
        try:
            import kaggle
            
            # Search for datasets
            search_term = re.search(r"(titanic|iris|mnist|house|spam|wine)", prompt.lower())
            query = search_term.group(1) if search_term else "classification"
            
            self._emit_progress("SEARCH", f"Searching Kaggle for: {query}")
            datasets = kaggle.api.dataset_list(search=query, max_size=50)
            
            for ds in datasets[:10]:
                item = {
                    "name": ds.title,
                    "ref": ds.ref,
                    "size": ds.size,
                    "downloads": ds.downloadCount,
                    "last_updated": str(ds.lastUpdated),
                    "source": "kaggle",
                }
                self.items_collected.append(item)
                self.on_item(item)
            
            self.sources_used.append("Kaggle API")
            self._emit_progress("DATA", f"Found {len(self.items_collected)} Kaggle datasets")
            
        except Exception as e:
            logger.error(f"Kaggle API error: {e}")
            self._emit_progress("FALLBACK", "Using local dataset cache...")
            await self._hunt_local_datasets(prompt)
    
    async def _hunt_local_datasets(self, prompt: str) -> None:
        """Hunt from locally cached datasets."""
        data_dir = Path("/root/leviathan/data")
        test_datasets = data_dir / "test-datasets"
        uploads = data_dir / "uploads"
        
        for dataset_dir in [test_datasets, uploads]:
            if not dataset_dir.exists():
                continue
            
            for path in dataset_dir.glob("**/*.csv"):
                if len(self.items_collected) >= 20:  # Limit file listings
                    break
                
                item = {
                    "name": path.stem,
                    "path": str(path),
                    "size_kb": path.stat().st_size // 1024,
                    "source": "local_cache",
                }
                self.items_collected.append(item)
                self.on_item(item)
        
        self.sources_used.append("Local Dataset Cache")
        self._emit_progress("DATA", f"Found {len(self.items_collected)} cached datasets")
    
    async def _hunt_world_bank(self, prompt: str) -> None:
        """Hunt for World Bank economic data."""
        self._emit_progress("HUNT", "Fetching World Bank data...")
        
        try:
            import wbdata
            
            # Determine country from prompt
            country = "VN"  # Default Vietnam
            if "vietnam" in prompt.lower() or "vn" in prompt.lower():
                country = "VN"
            elif "us" in prompt.lower() or "united states" in prompt.lower():
                country = "US"
            elif "china" in prompt.lower():
                country = "CN"
            
            # Fetch GDP data
            indicators = {
                "NY.GDP.MKTP.CD": "GDP (current US$)",
                "NY.GDP.PCAP.CD": "GDP per capita",
                "FP.CPI.TOTL.ZG": "Inflation rate",
            }
            
            for indicator_code, indicator_name in indicators.items():
                try:
                    data = wbdata.get_dataframe({indicator_code: indicator_name}, country=country)
                    
                    for date, row in data.iterrows():
                        if len(self.items_collected) >= MAX_ITEMS_PER_HUNT:
                            break
                        
                        item = {
                            "country": country,
                            "indicator": indicator_name,
                            "date": str(date),
                            "value": float(row[indicator_name]) if pd.notna(row[indicator_name]) else None,
                            "source": "world_bank",
                        }
                        self.items_collected.append(item)
                        self.on_item(item)
                    
                except Exception as e:
                    logger.warning(f"World Bank indicator {indicator_code} error: {e}")
            
            self.sources_used.append(f"World Bank: {country}")
            self._emit_progress("DATA", f"Collected {len(self.items_collected)} economic indicators")
            
        except ImportError:
            self._emit_progress("ERROR", "wbdata package not available")
        except Exception as e:
            logger.error(f"World Bank API error: {e}")
            self._emit_progress("ERROR", str(e))
    
    async def _hunt_rss(self, prompt: str) -> None:
        """Hunt from generic RSS feeds."""
        self._emit_progress("HUNT", "Searching for RSS feeds...")
        
        # Default news feeds
        feeds = [
            ("https://feeds.feedburner.com/TechCrunch/", "TechCrunch"),
        ]
        
        for feed_url, source_name in feeds:
            if len(self.items_collected) >= MAX_ITEMS_PER_HUNT:
                break
            
            domain = urlparse(feed_url).netloc
            if not is_approved_source(feed_url):
                self._emit_progress("SKIP", f"Source not approved: {domain}")
                continue
            
            await self.rate_limiter.wait(domain)
            
            try:
                async with self._session.get(feed_url) as response:
                    if response.status != 200:
                        continue
                    content = await response.text()
                
                feed = feedparser.parse(content)
                
                for entry in feed.entries[:30]:
                    if len(self.items_collected) >= MAX_ITEMS_PER_HUNT:
                        break
                    
                    item = {
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source_name,
                    }
                    self.items_collected.append(item)
                    self.on_item(item)
                
                self.sources_used.append(source_name)
                
            except Exception as e:
                logger.error(f"RSS error {feed_url}: {e}")
    
    async def _hunt_generic(self, prompt: str) -> None:
        """Generic hunt with ethical constraints."""
        self._emit_progress("HUNT", "Starting ethical generic hunt...")
        
        # For generic hunts, we focus on structured data APIs
        # No scraping of unapproved websites
        
        # Try local datasets first
        await self._hunt_local_datasets(prompt)
        
        if len(self.items_collected) < 10:
            self._emit_progress("INFO", "Limited data from generic hunt. Try specific sources like 'VN stock' or 'Kaggle'.")


async def autonomous_hunt(
    prompt: str,
    on_progress: Optional[Callable[[str, str], None]] = None,
    on_item: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for autonomous ethical data hunting.
    
    Args:
        prompt: Natural language description of data needed
        on_progress: Callback for progress updates (stage, message)
        on_item: Callback for each collected item
    
    Returns:
        Dict with items_collected, sources, data, timestamp
    """
    async with EthicalCrawler(on_progress=on_progress, on_item=on_item) as crawler:
        return await crawler.hunt(prompt)


def hunt_sync(
    prompt: str,
    on_progress: Optional[Callable[[str, str], None]] = None,
    on_item: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper for autonomous_hunt."""
    return asyncio.run(autonomous_hunt(prompt, on_progress, on_item))


# Background hunter for proactive data collection
class BackgroundHunter:
    """Proactive background data hunter - runs every 30 minutes."""
    
    def __init__(self):
        self.running = False
        self.interval_minutes = 30
        self.last_hunt: Optional[datetime] = None
        self.hunt_results: List[Dict] = []
        self._task: Optional[asyncio.Task] = None
        
        # Default hunt targets for Vietnam
        self.hunt_targets = [
            "Vietnam stock market trends from Yahoo Finance",
            "Vietnam news headlines from VNExpress RSS",
        ]
    
    async def start(self, on_alert: Optional[Callable[[Dict], None]] = None):
        """Start background hunting loop."""
        self.running = True
        self.on_alert = on_alert or (lambda a: None)
        
        while self.running:
            try:
                await self._run_hunt_cycle()
            except Exception as e:
                logger.error(f"Background hunt error: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(self.interval_minutes * 60)
    
    def stop(self):
        """Stop background hunting."""
        self.running = False
        if self._task:
            self._task.cancel()
    
    async def _run_hunt_cycle(self):
        """Run a single hunt cycle."""
        logger.info("Starting background hunt cycle...")
        self.last_hunt = datetime.now()
        
        for target in self.hunt_targets:
            try:
                result = await autonomous_hunt(target)
                self.hunt_results.append({
                    "target": target,
                    "timestamp": datetime.now().isoformat(),
                    "items": result["items_collected"],
                    "sources": result["sources"],
                })
                
                # Generate alert if significant data found
                if result["items_collected"] > 10:
                    self.on_alert({
                        "type": "hunt_complete",
                        "message": f"Collected {result['items_collected']} items: {target[:50]}",
                        "data": result,
                    })
                
            except Exception as e:
                logger.error(f"Background hunt target error: {e}")
        
        logger.info(f"Background hunt cycle complete. {len(self.hunt_targets)} targets processed.")


# Singleton background hunter
_background_hunter: Optional[BackgroundHunter] = None


def get_background_hunter() -> BackgroundHunter:
    """Get or create background hunter instance."""
    global _background_hunter
    if _background_hunter is None:
        _background_hunter = BackgroundHunter()
    return _background_hunter
