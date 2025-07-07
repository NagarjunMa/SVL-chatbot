"""
Content Filtering and Rate Limiting for SVL Chatbot
Advanced content moderation, abuse prevention, and rate limiting system
"""

import re
import time
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from functools import wraps

from utils.logger import get_logger
from utils.security_core import SecurityContext, SecurityLevel

logger = get_logger("content_filter")

class ContentThreatLevel(Enum):
    """Content threat levels"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    BLOCKED = "blocked"

class AbuseTrigger(Enum):
    """Types of abuse triggers"""
    RATE_LIMIT = "rate_limit"
    SPAM_DETECTED = "spam_detected"
    PROFANITY = "profanity"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    REPETITIVE_BEHAVIOR = "repetitive_behavior"
    SUSPICIOUS_PATTERNS = "suspicious_patterns"
    THREAT_DETECTED = "threat_detected"

@dataclass
class ContentAnalysisResult:
    """Result of content analysis"""
    threat_level: ContentThreatLevel
    is_blocked: bool
    confidence_score: float
    detected_issues: List[str] = field(default_factory=list)
    flagged_content: Dict[str, List[str]] = field(default_factory=dict)
    recommended_action: str = ""
    filter_reason: Optional[str] = None

@dataclass
class RateLimitResult:
    """Result of rate limiting check"""
    is_allowed: bool
    current_count: int
    limit: int
    reset_time: datetime
    retry_after: Optional[int] = None
    warning_message: Optional[str] = None

@dataclass
class UserBehaviorProfile:
    """User behavior tracking profile"""
    user_id: str
    total_requests: int = 0
    blocked_requests: int = 0
    spam_score: float = 0.0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    suspicious_patterns: List[str] = field(default_factory=list)
    warning_count: int = 0
    is_flagged: bool = False
    reputation_score: float = 1.0  # 0.0 to 1.0

class ContentModerator:
    """Advanced content moderation system"""
    
    def __init__(self):
        # Profanity and inappropriate content patterns
        self.profanity_patterns = [
            # Common profanity (masked for security)
            r'\b(damn|hell|crap|stupid|idiot)\b',
            # Add more patterns as needed (this is a basic example)
        ]
        
        # Threat and violence patterns
        self.threat_patterns = [
            r'\b(kill|murder|bomb|shoot|attack|destroy|harm|hurt|violence)\b',
            r'\b(threat|terroris[mt]|explosive|weapon|gun|knife)\b',
            r'\b(die|death|suicide|self[-\s]harm)\b',
        ]
        
        # Spam patterns
        self.spam_patterns = [
            r'(https?://[^\s]+){3,}',  # Multiple URLs
            r'(\b\w+\b.*?){20,}',      # Very repetitive text
            r'[A-Z]{10,}',             # Excessive caps
            r'(.)\1{10,}',             # Character repetition
            r'\$\d+|\bmoney\b|\bfree\b|\bwin\b|\bprize\b',  # Promotional spam
        ]
        
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(sex|porn|adult|explicit|nsfw)\b',
            r'\b(drug|cocaine|marijuana|weed|alcohol)\b',
            r'\b(gambling|casino|bet|poker)\b',
        ]
        
        # Vehicle theft simulation patterns (for context)
        self.valid_theft_keywords = [
            'stolen', 'theft', 'missing', 'lost', 'vehicle', 'car', 'truck',
            'motorcycle', 'report', 'police', 'insurance', 'recovery'
        ]
    
    def analyze_content(self, text: str, context: SecurityContext) -> ContentAnalysisResult:
        """Comprehensive content analysis"""
        threat_level = ContentThreatLevel.SAFE
        detected_issues = []
        flagged_content = {}
        confidence_score = 0.0
        
        # Check for profanity
        profanity_matches = self._check_patterns(text, self.profanity_patterns)
        if profanity_matches:
            detected_issues.append("Profanity detected")
            flagged_content["profanity"] = profanity_matches
            threat_level = ContentThreatLevel.LOW_RISK
            confidence_score = max(confidence_score, 0.3)
        
        # Check for threats and violence
        threat_matches = self._check_patterns(text, self.threat_patterns)
        if threat_matches:
            detected_issues.append("Potential threats detected")
            flagged_content["threats"] = threat_matches
            threat_level = ContentThreatLevel.HIGH_RISK
            confidence_score = max(confidence_score, 0.8)
        
        # Check for spam
        spam_matches = self._check_patterns(text, self.spam_patterns)
        if spam_matches:
            detected_issues.append("Spam patterns detected")
            flagged_content["spam"] = spam_matches
            threat_level = max(threat_level, ContentThreatLevel.MEDIUM_RISK)
            confidence_score = max(confidence_score, 0.6)
        
        # Check for inappropriate content
        inappropriate_matches = self._check_patterns(text, self.inappropriate_patterns)
        if inappropriate_matches:
            detected_issues.append("Inappropriate content detected")
            flagged_content["inappropriate"] = inappropriate_matches
            threat_level = max(threat_level, ContentThreatLevel.MEDIUM_RISK)
            confidence_score = max(confidence_score, 0.5)
        
        # Context-aware analysis
        if not self._is_relevant_to_vehicle_theft(text):
            if len(text) > 100:  # Only flag longer off-topic messages
                detected_issues.append("Content appears off-topic for vehicle theft reporting")
                threat_level = max(threat_level, ContentThreatLevel.LOW_RISK)
                confidence_score = max(confidence_score, 0.2)
        
        # Determine if content should be blocked
        is_blocked = threat_level in [ContentThreatLevel.HIGH_RISK, ContentThreatLevel.BLOCKED]
        
        # Generate recommended action
        recommended_action = self._get_recommended_action(threat_level, detected_issues)
        
        # Generate filter reason if blocked
        filter_reason = None
        if is_blocked:
            filter_reason = f"Content blocked due to: {', '.join(detected_issues)}"
        
        return ContentAnalysisResult(
            threat_level=threat_level,
            is_blocked=is_blocked,
            confidence_score=confidence_score,
            detected_issues=detected_issues,
            flagged_content=flagged_content,
            recommended_action=recommended_action,
            filter_reason=filter_reason
        )
    
    def _check_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Check text against pattern list"""
        matches = []
        text_lower = text.lower()
        
        for pattern in patterns:
            found = re.findall(pattern, text_lower, re.IGNORECASE)
            matches.extend(found)
        
        return list(set(matches))  # Remove duplicates
    
    def _is_relevant_to_vehicle_theft(self, text: str) -> bool:
        """Check if content is relevant to vehicle theft reporting"""
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.valid_theft_keywords if keyword in text_lower)
        return keyword_count >= 1 or len(text) < 50  # Allow short messages
    
    def _get_recommended_action(self, threat_level: ContentThreatLevel, issues: List[str]) -> str:
        """Get recommended action based on threat level"""
        if threat_level == ContentThreatLevel.BLOCKED:
            return "Block content immediately and flag user"
        elif threat_level == ContentThreatLevel.HIGH_RISK:
            return "Block content and review user behavior"
        elif threat_level == ContentThreatLevel.MEDIUM_RISK:
            return "Flag for review and warn user"
        elif threat_level == ContentThreatLevel.LOW_RISK:
            return "Log incident and provide gentle warning"
        else:
            return "Allow content"

class RateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self):
        self.limits = {
            SecurityLevel.LOW: {
                "requests_per_minute": 20,
                "requests_per_hour": 200,
                "requests_per_day": 1000
            },
            SecurityLevel.MEDIUM: {
                "requests_per_minute": 15,
                "requests_per_hour": 150,
                "requests_per_day": 800
            },
            SecurityLevel.HIGH: {
                "requests_per_minute": 10,
                "requests_per_hour": 100,
                "requests_per_day": 500
            },
            SecurityLevel.CRITICAL: {
                "requests_per_minute": 5,
                "requests_per_hour": 50,
                "requests_per_day": 200
            }
        }
        
        # Track requests per user
        self.user_requests = defaultdict(lambda: {
            "minute": deque(),
            "hour": deque(),
            "day": deque()
        })
        
        # Thread lock for thread safety
        self._lock = threading.Lock()
    
    def check_rate_limit(self, user_id: str, security_level: SecurityLevel) -> RateLimitResult:
        """Check if user is within rate limits"""
        with self._lock:
            now = datetime.now(timezone.utc)
            limits = self.limits[security_level]
            
            # Clean old requests
            self._clean_old_requests(user_id, now)
            
            # Get current counts
            minute_count = len(self.user_requests[user_id]["minute"])
            hour_count = len(self.user_requests[user_id]["hour"])
            day_count = len(self.user_requests[user_id]["day"])
            
            # Check limits
            if minute_count >= limits["requests_per_minute"]:
                return RateLimitResult(
                    is_allowed=False,
                    current_count=minute_count,
                    limit=limits["requests_per_minute"],
                    reset_time=now + timedelta(minutes=1),
                    retry_after=60,
                    warning_message="Too many requests per minute. Please slow down."
                )
            
            if hour_count >= limits["requests_per_hour"]:
                return RateLimitResult(
                    is_allowed=False,
                    current_count=hour_count,
                    limit=limits["requests_per_hour"],
                    reset_time=now + timedelta(hours=1),
                    retry_after=3600,
                    warning_message="Hourly rate limit exceeded. Please try again later."
                )
            
            if day_count >= limits["requests_per_day"]:
                return RateLimitResult(
                    is_allowed=False,
                    current_count=day_count,
                    limit=limits["requests_per_day"],
                    reset_time=now + timedelta(days=1),
                    retry_after=86400,
                    warning_message="Daily rate limit exceeded. Please try again tomorrow."
                )
            
            # Add request to tracking
            self.user_requests[user_id]["minute"].append(now)
            self.user_requests[user_id]["hour"].append(now)
            self.user_requests[user_id]["day"].append(now)
            
            return RateLimitResult(
                is_allowed=True,
                current_count=minute_count + 1,
                limit=limits["requests_per_minute"],
                reset_time=now + timedelta(minutes=1)
            )
    
    def _clean_old_requests(self, user_id: str, now: datetime):
        """Clean old request timestamps"""
        user_data = self.user_requests[user_id]
        
        # Clean minute data
        while user_data["minute"] and now - user_data["minute"][0] > timedelta(minutes=1):
            user_data["minute"].popleft()
        
        # Clean hour data
        while user_data["hour"] and now - user_data["hour"][0] > timedelta(hours=1):
            user_data["hour"].popleft()
        
        # Clean day data
        while user_data["day"] and now - user_data["day"][0] > timedelta(days=1):
            user_data["day"].popleft()

class BehaviorAnalyzer:
    """User behavior analysis and abuse detection"""
    
    def __init__(self):
        self.user_profiles = {}
        self._lock = threading.Lock()
        
        # Behavior thresholds
        self.spam_threshold = 0.7
        self.reputation_threshold = 0.3
        self.warning_threshold = 3
    
    def analyze_user_behavior(self, user_id: str, content: str, context: SecurityContext) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        with self._lock:
            profile = self._get_or_create_profile(user_id)
            
            # Update activity
            profile.last_activity = datetime.now(timezone.utc)
            profile.total_requests += 1
            
            # Analyze current request
            behavior_flags = []
            
            # Check for repetitive content
            if self._is_repetitive_content(user_id, content):
                behavior_flags.append("repetitive_content")
                profile.spam_score = min(profile.spam_score + 0.1, 1.0)
            
            # Check for rapid requests
            if self._is_rapid_requests(user_id):
                behavior_flags.append("rapid_requests")
                profile.spam_score = min(profile.spam_score + 0.2, 1.0)
            
            # Check for suspicious patterns
            suspicious_patterns = self._detect_suspicious_patterns(content)
            if suspicious_patterns:
                behavior_flags.extend(suspicious_patterns)
                profile.suspicious_patterns.extend(suspicious_patterns)
                profile.spam_score = min(profile.spam_score + 0.15, 1.0)
            
            # Update reputation
            self._update_reputation(profile, behavior_flags)
            
            # Check if user should be flagged
            if profile.spam_score >= self.spam_threshold or profile.reputation_score <= self.reputation_threshold:
                profile.is_flagged = True
                behavior_flags.append("user_flagged")
            
            return {
                "behavior_flags": behavior_flags,
                "spam_score": profile.spam_score,
                "reputation_score": profile.reputation_score,
                "is_flagged": profile.is_flagged,
                "warning_count": profile.warning_count,
                "profile": profile
            }
    
    def _get_or_create_profile(self, user_id: str) -> UserBehaviorProfile:
        """Get or create user behavior profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
        return self.user_profiles[user_id]
    
    def _is_repetitive_content(self, user_id: str, content: str) -> bool:
        """Check for repetitive content patterns"""
        # Simple implementation - check if user sent similar content recently
        profile = self.user_profiles.get(user_id)
        if not profile:
            return False
        
        # Store recent messages (simplified for POC)
        if not hasattr(profile, 'recent_messages'):
            profile.recent_messages = deque(maxlen=10)
        
        # Check for similar content
        for msg in profile.recent_messages:
            if self._similarity_score(content, msg) > 0.8:
                return True
        
        profile.recent_messages.append(content)
        return False
    
    def _is_rapid_requests(self, user_id: str) -> bool:
        """Check for rapid request patterns"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return False
        
        if not hasattr(profile, 'request_timestamps'):
            profile.request_timestamps = deque(maxlen=20)
        
        now = datetime.now(timezone.utc)
        profile.request_timestamps.append(now)
        
        # Check if more than 5 requests in last 30 seconds
        recent_count = sum(1 for ts in profile.request_timestamps 
                          if now - ts < timedelta(seconds=30))
        
        return recent_count > 5
    
    def _detect_suspicious_patterns(self, content: str) -> List[str]:
        """Detect suspicious patterns in content"""
        patterns = []
        
        # Check for encoded content
        if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', content):
            patterns.append("encoded_content")
        
        # Check for unusual character distribution
        if len(set(content)) / len(content) > 0.7 and len(content) > 20:
            patterns.append("unusual_characters")
        
        # Check for very long words
        words = content.split()
        if any(len(word) > 30 for word in words):
            patterns.append("unusual_word_length")
        
        return patterns
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_reputation(self, profile: UserBehaviorProfile, behavior_flags: List[str]):
        """Update user reputation based on behavior"""
        # Decrease reputation for negative behavior
        penalty = len(behavior_flags) * 0.05
        profile.reputation_score = max(profile.reputation_score - penalty, 0.0)
        
        # Slowly increase reputation for good behavior
        if not behavior_flags:
            profile.reputation_score = min(profile.reputation_score + 0.01, 1.0)

class AbusePreventionSystem:
    """Comprehensive abuse prevention system"""
    
    def __init__(self):
        self.content_moderator = ContentModerator()
        self.rate_limiter = RateLimiter()
        self.behavior_analyzer = BehaviorAnalyzer()
        self._lock = threading.Lock()
    
    def process_request(self, user_input: str, context: SecurityContext) -> Dict[str, Any]:
        """Process request through all abuse prevention systems"""
        with self._lock:
            results = {
                "allowed": True,
                "warnings": [],
                "errors": [],
                "actions_taken": [],
                "metadata": {}
            }
            
            # 1. Rate limiting check
            rate_limit_result = self.rate_limiter.check_rate_limit(
                context.user_id, context.security_level
            )
            
            if not rate_limit_result.is_allowed:
                results["allowed"] = False
                results["errors"].append(rate_limit_result.warning_message)
                results["actions_taken"].append("rate_limited")
                results["metadata"]["rate_limit"] = rate_limit_result
                return results
            
            # 2. Content analysis
            content_analysis = self.content_moderator.analyze_content(user_input, context)
            results["metadata"]["content_analysis"] = content_analysis
            
            if content_analysis.is_blocked:
                results["allowed"] = False
                results["errors"].append(content_analysis.filter_reason)
                results["actions_taken"].append("content_blocked")
                return results
            
            # Add warnings for lower-level threats
            if content_analysis.threat_level != ContentThreatLevel.SAFE:
                results["warnings"].extend(content_analysis.detected_issues)
                results["actions_taken"].append("content_flagged")
            
            # 3. Behavior analysis
            behavior_analysis = self.behavior_analyzer.analyze_user_behavior(
                context.user_id, user_input, context
            )
            results["metadata"]["behavior_analysis"] = behavior_analysis
            
            # Handle flagged users
            if behavior_analysis["is_flagged"]:
                if behavior_analysis["spam_score"] > 0.9:
                    results["allowed"] = False
                    results["errors"].append("Account flagged for suspicious activity")
                    results["actions_taken"].append("user_blocked")
                    return results
                else:
                    results["warnings"].append("Unusual activity detected")
                    results["actions_taken"].append("user_warned")
            
            return results

# Decorator for automatic abuse prevention
def abuse_protection(security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Decorator to add abuse protection to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context from function arguments
            context = None
            user_input = None
            
            # Try to find SecurityContext and user input in arguments
            for arg in args:
                if isinstance(arg, SecurityContext):
                    context = arg
                elif isinstance(arg, str) and len(arg) > 0:
                    user_input = arg
            
            # Check kwargs as well
            for key, value in kwargs.items():
                if key == 'context' and isinstance(value, SecurityContext):
                    context = value
                elif key in ['user_input', 'message', 'text'] and isinstance(value, str):
                    user_input = value
            
            if not context or not user_input:
                logger.warning("Abuse protection decorator: Missing context or user input")
                return func(*args, **kwargs)
            
            # Apply abuse prevention
            abuse_system = AbusePreventionSystem()
            result = abuse_system.process_request(user_input, context)
            
            if not result["allowed"]:
                logger.warning(f"Request blocked by abuse prevention: {result['errors']}")
                raise ValueError(f"Request blocked: {'; '.join(result['errors'])}")
            
            if result["warnings"]:
                logger.info(f"Request warnings: {result['warnings']}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global abuse prevention system instance
global_abuse_prevention = AbusePreventionSystem()

def check_content_safety(content: str, context: SecurityContext) -> ContentAnalysisResult:
    """Check content safety"""
    return global_abuse_prevention.content_moderator.analyze_content(content, context)

def check_rate_limits(user_id: str, security_level: SecurityLevel) -> RateLimitResult:
    """Check rate limits for user"""
    return global_abuse_prevention.rate_limiter.check_rate_limit(user_id, security_level)

def analyze_behavior(user_id: str, content: str, context: SecurityContext) -> Dict[str, Any]:
    """Analyze user behavior"""
    return global_abuse_prevention.behavior_analyzer.analyze_user_behavior(user_id, content, context) 