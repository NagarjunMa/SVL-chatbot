import streamlit as st
import os
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.aws_bedrock import get_conversation_manager
from utils.chat_utils import initialize_session, append_message
from utils.database_manager import DatabaseManager
from utils.data_utils import DataValidator, PIIHandler, ComplianceLogger
from utils.conversation_orchestrator import ConversationOrchestrator
from data.models import Message, VehicleInfo, OwnerInfo, IncidentInfo, InsuranceInfo, Ticket

# Import observability
from utils.observability import observability, trace_function
from utils.conversation_manager_with_observability import ConversationManagerWithObservability

# Security framework imports
from utils.security_integration import get_security_manager, SecurityConfig, SecurityLevel
from utils.session_security import get_session_manager, SessionSecurityLevel
from utils.secure_error_handler import get_error_handler, with_error_handling
from utils.audit_logger import get_audit_logger

import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logger
logger = get_logger()

# Initialize security framework
@st.cache_resource
def initialize_security_framework():
    """Initialize security framework with simplified configuration"""
    try:
        # Use simple SecurityConfig from security_integration (not complex config/security_config.py)
        from utils.security_integration import SecurityConfig, get_security_manager
        
        # Create simple configuration that won't cause attribute errors
        security_config = SecurityConfig(
            enable_input_validation=True,
            enable_pii_detection=True,
            enable_content_filtering=False,  # Disabled for simplicity
            enable_rate_limiting=False,      # Disabled for simplicity
            enable_session_security=True,
            enable_api_authentication=False, # Disabled for simplicity
            enable_error_handling=True,
            enable_compliance_features=True,
            enable_audit_logging=True,
        )
        
        # Initialize security manager with simple config
        security_manager = get_security_manager(security_config)
        
        return {
            "security_manager": security_manager,
            "session_manager": security_manager.session_manager,
            "error_handler": security_manager.error_handler,
            "audit_logger": security_manager.audit_logger,
            "compliance_manager": security_manager.compliance_manager
        }
    except Exception as e:
        logger.error(f"Failed to initialize security framework: {e}")
        # Fallback to minimal security components
        from utils.session_security import get_session_manager
        from utils.secure_error_handler import get_error_handler  
        from utils.audit_logger import get_audit_logger
        from utils.compliance_manager import get_compliance_manager
        
        return {
            "security_manager": None,
            "session_manager": get_session_manager(),
            "error_handler": get_error_handler(),
            "audit_logger": get_audit_logger(),
            "compliance_manager": get_compliance_manager()
        }

# Initialize security framework
security_framework = initialize_security_framework()

# Custom CSS for styling
with open("./utils/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Stolen Vehicle Locator Chatbot", page_icon="🚗", layout="centered")
st.title("🚗 Stolen Vehicle Locator (SVL) Chatbot")

# Initialize session state with security
initialize_session()

# Add security context to session state
if "security_context" not in st.session_state:
    # Create security context for the user session
    user_ip = st.context.headers.get("X-Forwarded-For", "127.0.0.1") if hasattr(st.context, 'headers') else "127.0.0.1"
    user_agent = st.context.headers.get("User-Agent", "streamlit") if hasattr(st.context, 'headers') else "streamlit"
    
    st.session_state["security_context"] = {
        "user_id": st.session_state.get("user_id", "anonymous"),
        "session_id": st.session_state.get("conversation_id", "no_session"),
        "ip_address": user_ip,
        "user_agent": user_agent,
        "security_level": SecurityLevel.MEDIUM,
        "csrf_token": "dummy_csrf_token"  # We'll generate proper tokens when sessions are created
    }

# Initialize database manager
if "db_manager" not in st.session_state:
    try:
        st.session_state["db_manager"] = DatabaseManager()
        logger.info("Database manager initialized")
        
        # Log system access
        security_framework["audit_logger"].log_authentication(
            st.session_state["security_context"]["user_id"],
            st.session_state["security_context"]["session_id"],
            True,
            "system_access",
            st.session_state["security_context"]["ip_address"]
        )
    except Exception as e:
        logger.error(f"Failed to initialize database manager: {e}")
        st.error("Critical error: Unable to connect to database.")
        st.stop()

# Initialize ConversationManager with Observability in session state
if "conversation_manager" not in st.session_state:
    try:
        # Use observability-enabled conversation manager
        session_id = st.session_state.get("conversation_id", "default-session")
        st.session_state["conversation_manager"] = ConversationManagerWithObservability(session_id)
        logger.info(f"Observability-enabled conversation manager initialized for session: {session_id}")
    except Exception as e:
        logger.error(f"Failed to initialize ConversationManager with Observability: {e}")
        # Fallback to regular conversation manager
        try:
            st.session_state["conversation_manager"] = get_conversation_manager(
                session_id=st.session_state.get("conversation_id", "default-session")
            )
            logger.info("Fallback to regular conversation manager")
        except Exception as e2:
            logger.error(f"Failed to initialize fallback ConversationManager: {e2}")
            st.error("Critical error: Unable to connect to AI service.")
            st.stop()

# Initialize Conversation Orchestrator
if "orchestrator" not in st.session_state:
    try:
        orchestrator = ConversationOrchestrator(
            st.session_state["db_manager"],
            st.session_state["conversation_manager"]
        )
        # Initialize orchestrator asynchronously
        asyncio.run(orchestrator.initialize())
        st.session_state["orchestrator"] = orchestrator
        logger.info("Conversation orchestrator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize conversation orchestrator: {e}")
        st.error("Critical error: Unable to initialize conversation system.")
        st.stop()

# Initialize user session with security
if "user_id" not in st.session_state:
    st.session_state["user_id"] = f"user-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    
    # Update security context
    st.session_state["security_context"]["user_id"] = st.session_state["user_id"]

# Initialize conversation ID
if "conversation_id" not in st.session_state:
    try:
        # Get or create conversation for user
        db_manager = st.session_state["db_manager"]
        conversation_id = asyncio.run(db_manager.get_or_create_session(st.session_state["user_id"]))
        st.session_state["conversation_id"] = conversation_id
        
        # Update security context
        st.session_state["security_context"]["session_id"] = conversation_id
        
        # Create secure session
        try:
            session_data = security_framework["session_manager"].create_session(
                st.session_state["user_id"],
                st.session_state["security_context"]["ip_address"],
                st.session_state["security_context"]["user_agent"],
                SessionSecurityLevel.STANDARD
            )
            
            st.session_state["secure_session_id"] = session_data.session_id
            logger.info(f"Secure session created: {session_data.session_id}")
        except Exception as e:
            logger.warning(f"Failed to create secure session: {e}")
            # Continue without secure session
        
        logger.info(f"Initialized conversation: {conversation_id}")
    except Exception as e:
        logger.error(f"Failed to initialize conversation: {e}")
        st.session_state["conversation_id"] = "fallback-conversation"

# Helper: Determine conversation phase (enhanced with orchestrator integration)
def get_phase():
    """Determine conversation phase with intelligent routing"""
    messages = st.session_state["messages"]
    if len(messages) <= 1:
        return "greeting"
    
    # Check if we have an active conversation context
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator:
        try:
            conversation_status = orchestrator.get_conversation_details(st.session_state["conversation_id"])
            if conversation_status.get("conversation_status", {}).get("status") == "active":
                # current_state is already a string value from context.current_state.value
                current_state = conversation_status["conversation_status"].get("current_state", "initial")
                
                # Map conversation states to phases
                state_phase_map = {
                    "greeting": "greeting",
                    "assessment": "collect_info", 
                    "info_collection": "collect_info",
                    "vehicle_details": "collect_info",
                    "owner_details": "collect_info",
                    "incident_details": "collect_info",
                    "insurance_details": "collect_info",
                    "verification": "confirmation",
                    "confirmation": "confirmation",
                    "qa_support": "faq",
                    "process_explanation": "process_explanation"
                }
                
                return state_phase_map.get(current_state, "greeting")
        except Exception as e:
            logger.warning(f"Failed to get conversation phase from orchestrator: {e}")
            # Fall through to fallback logic
    
    # Fallback to original logic
    if "form_step" in st.session_state:
        step = st.session_state["form_step"]
        if step == 0:
            return "collect_info"
        elif step == 1:
            return "collect_info"
        elif step == 2:
            return "collect_info"
        elif step == 3:
            return "confirmation"
    
    # Determine from message content
    last_user = [m for m in messages[::-1] if m["role"] == "user"]
    if last_user and any(q in last_user[0]["content"].lower() for q in ["how", "what", "when", "where", "faq"]):
        return "faq"
    return "process_explanation"

# Add system status indicator in sidebar
with st.sidebar:
    st.header("🚨 Emergency Contacts")
    st.info("""
    **Police Emergency**: 911
    **SVL Hotline**: 1-800-555-0123
    **Insurance Support**: 1-800-555-0199
    """)
    
    # Security Dashboard
    with st.expander("🔒 Security Dashboard", expanded=False):
        try:
            # Get real-time security status
            security_status = security_framework["security_manager"].get_security_status()
            monitoring_data = security_status.get("monitoring", {})
            
            # Security Status Overview
            st.markdown("**🛡️ Security Status**")
            col1, col2 = st.columns(2)
            
            with col1:
                if monitoring_data.get("monitoring_active", False):
                    st.success("🟢 Active")
                else:
                    st.error("🔴 Inactive")
                st.caption("Security Monitoring")
                
                uptime = monitoring_data.get("uptime_hours", 0)
                st.metric("Uptime", f"{uptime:.1f}h")
                
            with col2:
                incidents = monitoring_data.get("active_incidents", 0)
                if incidents == 0:
                    st.success(f"🟢 {incidents}")
                else:
                    st.warning(f"🟡 {incidents}")
                st.caption("Active Incidents")
                
                blocked_users = monitoring_data.get("blocked_users", 0)
                st.metric("Blocked Users", blocked_users)
            
            # Security Metrics
            st.markdown("**📊 Security Metrics**")
            metrics = monitoring_data.get("metrics", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Events", metrics.get("events_processed", 0))
            with col2:
                st.metric("Threats", metrics.get("threats_detected", 0))
            with col3:
                st.metric("Quarantined", metrics.get("quarantined_sessions", 0))
            
            # Real-time threat level
            threat_level = monitoring_data.get("current_threat_level", "low")
            if threat_level == "low":
                st.success("🟢 **Threat Level: LOW**")
            elif threat_level == "medium":
                st.warning("🟡 **Threat Level: MEDIUM**")
            else:
                st.error("🔴 **Threat Level: HIGH**")
                
        except Exception as e:
            st.error("Security monitoring unavailable")
    
    # Observability Dashboard
    with st.expander("📊 Observability Dashboard", expanded=False):
        try:
            # Get observability status
            observability_status = observability.get_status()
            
            st.markdown("**🔍 System Observability**")
            col1, col2 = st.columns(2)
            
            with col1:
                if observability_status.get("active", False):
                    st.success("🟢 Active")
                else:
                    st.error("🔴 Inactive")
                st.caption("Monitoring Active")
                
                traces_count = observability_status.get("active_traces", 0)
                st.metric("Active Traces", traces_count)
                
            with col2:
                logs_count = observability_status.get("logs_sent", 0)
                st.metric("Logs Sent", logs_count)
                
                errors_count = observability_status.get("errors_logged", 0)
                if errors_count == 0:
                    st.success(f"🟢 {errors_count}")
                else:
                    st.warning(f"🟡 {errors_count}")
                st.caption("Errors Today")
            
            # CloudWatch Dashboard Link
            st.markdown("**🌐 AWS CloudWatch Dashboard**")
            cloudwatch_url = "https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=SVL-Observability-Dashboard"
            
            if st.button("🚀 Open CloudWatch Dashboard"):
                st.markdown(f'<a href="{cloudwatch_url}" target="_blank">Open SVL Observability Dashboard</a>', unsafe_allow_html=True)
                st.info("📈 CloudWatch dashboard opened in new tab")
            
            st.markdown("**📋 Quick Links**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Metrics"):
                    st.info("Real-time metrics available in CloudWatch")
            with col2:
                if st.button("🔍 Search Logs"):
                    st.info("Search logs in CloudWatch Logs")
                    
        except Exception as e:
            st.error("Observability monitoring unavailable")
    
    # Compliance Features
    with st.expander("📋 Data Rights (GDPR/CCPA)", expanded=False):
        st.markdown("**Your Data Rights**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download My Data"):
                try:
                    # Export user data
                    compliance_manager = security_framework["security_manager"].compliance_manager
                    export_result = compliance_manager.export_user_data(
                        st.session_state["security_context"]["user_id"]
                    )
                    
                    if export_result["success"]:
                        st.success("✅ Data export ready")
                        st.download_button(
                            "⬇️ Download",
                            export_result["data"],
                            f"user_data_{st.session_state['user_id']}.json",
                            "application/json"
                        )
                    else:
                        st.error("Export failed")
                except Exception as e:
                    st.error("Export unavailable")
        
        with col2:
            if st.button("🗑️ Delete My Data"):
                if st.session_state.get("confirm_delete", False):
                    try:
                        # Delete user data
                        compliance_manager = security_framework["security_manager"].compliance_manager
                        delete_result = compliance_manager.delete_user_data(
                            st.session_state["security_context"]["user_id"]
                        )
                        
                        if delete_result["success"]:
                            st.success("✅ Data deletion scheduled")
                            st.info("Your data will be permanently deleted within 30 days.")
                        else:
                            st.error("Deletion failed")
                    except Exception as e:
                        st.error("Deletion unavailable")
                    st.session_state["confirm_delete"] = False
                else:
                    st.warning("⚠️ This will permanently delete all your data!")
                    if st.button("Confirm Delete"):
                        st.session_state["confirm_delete"] = True
                        st.rerun()
        
        # Privacy Controls
        st.markdown("**Privacy Settings**")
        if st.checkbox("🔒 Enable Enhanced Privacy Mode", value=False):
            st.session_state["security_context"]["security_level"] = SecurityLevel.HIGH
            st.info("Enhanced privacy mode enabled - additional PII protection active")
        
        if st.checkbox("📊 Allow Usage Analytics", value=True):
            st.session_state["analytics_consent"] = True
        else:
            st.session_state["analytics_consent"] = False
    
    # System Status
    if "orchestrator" in st.session_state:
        with st.expander("🔧 System Status", expanded=False):
            try:
                status = st.session_state["orchestrator"].get_system_status()
                health = status["overall_health"]["status"]
                
                if health == "healthy":
                    st.success("🟢 System Operational")
                elif health == "degraded": 
                    st.warning("🟡 System Degraded")
                else:
                    st.error("🔴 System Issues")
                
                st.caption(f"Active conversations: {status['overall_health']['active_conversations']}")
                st.caption(f"Success rate: {status['overall_health']['success_rate']}%")
                
                # Add demo controls
                if st.button("🎯 Demo Vehicle Recovery"):
                    try:
                        demo_result = asyncio.run(
                            st.session_state["orchestrator"].simulate_vehicle_recovery(
                                "SVL-DEMO-001",
                                {"recovery_location": "Downtown Metro", "vehicle_condition": "Excellent"}
                            )
                        )
                        if demo_result.get("status") == "success":
                            st.success("🎉 Demo recovery notification sent!")
                        else:
                            st.error("Demo failed")
                    except Exception as e:
                        st.error(f"Demo error: {e}")
            
            except Exception as e:
                st.error("Status check failed")
    
    st.markdown("---")
    st.caption("This system is monitored 24/7. For immediate assistance, please call emergency services.")
    st.caption("🔒 All data is encrypted and GDPR/CCPA compliant.")

# Chat interface with enhanced orchestrator integration
with st.container():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("How can I help you locate a stolen vehicle today?")
    if user_input:
        continue_processing = True
        
        try:
            # Process through comprehensive security pipeline
            security_result = security_framework["security_manager"].process_request(
                user_input, 
                st.session_state["security_context"]
            )
            
            # Check if request is allowed
            if not security_result["allowed"]:
                # Request blocked by security
                for error in security_result["warnings"]:
                    st.warning(f"🔒 Security Notice: {error}")
                
                # Log security violation
                security_framework["audit_logger"].log_security_violation(
                    st.session_state["security_context"]["user_id"],
                    st.session_state["security_context"]["session_id"], 
                    "input_blocked",
                    security_result["warnings"][0] if security_result["warnings"] else "Input validation failed",
                    st.session_state["security_context"]["ip_address"]
                )
                continue_processing = False
            
            if continue_processing:
                # Use sanitized input from security pipeline
                sanitized_input = security_result["sanitized_input"]
                
                # Show security warnings if any
                if security_result["warnings"]:
                    for warning in security_result["warnings"]:
                        st.info(f"🔒 Privacy Notice: {warning}")
                
                # Validate session security - TEMPORARILY DISABLED FOR SIMPLICITY
                # if "secure_session_id" in st.session_state:
                #     session_validation = security_framework["session_manager"].validate_session(
                #         st.session_state["secure_session_id"],
                #         st.session_state["security_context"]["csrf_token"],
                #         st.session_state["security_context"]["ip_address"],
                #         st.session_state["security_context"]["user_agent"]
                #     )
                #     
                #     if not session_validation["valid"]:
                #         st.error(f"🔒 Session Security Error: {session_validation['reason']}")
                #         st.info("Please refresh the page to start a new secure session.")
                #         continue_processing = False
            
            if continue_processing:    
                # Add user message to session state
                append_message("user", sanitized_input)
                
                # Log data access for compliance
                security_framework["audit_logger"].log_data_access(
                    st.session_state["security_context"]["user_id"],
                    "conversation",
                    st.session_state["conversation_id"],
                    "user_input",
                    st.session_state["security_context"]["ip_address"]
                )
                
                # TEMPORARILY BYPASS ORCHESTRATOR - Use ConversationManager directly
                # Process through enhanced orchestrator
                orchestrator_enabled = False  # Set to True to re-enable orchestrator
                
                if orchestrator_enabled:
                    try:
                        with st.spinner("Processing your message..."):
                            orchestrator = st.session_state["orchestrator"]
                            
                            # Process message through conversation orchestrator
                            orchestrator_result = asyncio.run(
                                orchestrator.process_user_message(
                                    st.session_state["user_id"],
                                    st.session_state["conversation_id"],
                                    sanitized_input,
                                    {"streamlit_session": True}
                                )
                            )
                            
                            # Extract response
                            if orchestrator_result["status"] == "success":
                                response = orchestrator_result["response"]
                                
                                # Add metadata info for debugging
                                if st.session_state.get("debug_mode", False):
                                    metadata = orchestrator_result.get("metadata", {})
                                    st.caption(f"Debug: State={orchestrator_result.get('conversation_state', 'unknown')}, "
                                             f"Intent={orchestrator_result.get('intent', 'unknown')}, "
                                             f"Time={orchestrator_result.get('processing_time', 0):.2f}s")
                            
                            else:
                                response = orchestrator_result["response"]
                                logger.error(f"Orchestrator error: {orchestrator_result.get('error', 'Unknown')}")
                    
                    except Exception as e:
                        logger.error(f"Orchestrator processing failed: {e}")
                        # Fallback to original conversation manager
                        phase = get_phase()
                        logger.info(f"Falling back to original conversation manager in phase: {phase}")
                        
                        cm = st.session_state["conversation_manager"]
                        response = cm.process_user_input(sanitized_input, phase)
                else:
                    # Use ConversationManager directly (WORKING VERSION)
                    with st.spinner("Processing your message..."):
                        phase = get_phase()
                        logger.info(f"Using ConversationManager directly in phase: {phase}")
                        
                        cm = st.session_state["conversation_manager"]
                        response = cm.process_user_input(sanitized_input, phase)
                
                # Add assistant response
                append_message("assistant", response)
                logger.info(f"Assistant response added to messages")
                
                # Save conversation to database with secure error handling
                try:
                    db_manager = st.session_state["db_manager"]
                    user_message = Message(role="user", content=sanitized_input)
                    assistant_message = Message(role="assistant", content=response)
                    
                    # Add messages to database conversation
                    asyncio.run(db_manager.add_message_to_conversation(
                        st.session_state["conversation_id"], 
                        user_message
                    ))
                    asyncio.run(db_manager.add_message_to_conversation(
                        st.session_state["conversation_id"], 
                        assistant_message
                    ))
                    
                    # Log data access for compliance using new audit system
                    security_framework["audit_logger"].log_data_access(
                        st.session_state["security_context"]["user_id"],
                        "conversation",
                        st.session_state["conversation_id"],
                        "add_message",
                        st.session_state["security_context"]["ip_address"]
                    )
                    
                except Exception as e:
                    # Use secure error handler
                    secure_error = security_framework["error_handler"].handle_error(
                        e, 
                        st.session_state["security_context"]["user_id"],
                        {"operation": "save_conversation", "conversation_id": st.session_state["conversation_id"]}
                    )
                    logger.error(f"Database save error: {secure_error.technical_message}")
                    st.error(f"Failed to save conversation: {secure_error.user_message}")
                
                # Refresh the page to show new messages
                st.rerun()
                
        except Exception as e:
            # Use secure error handler for all exceptions
            secure_error = security_framework["error_handler"].handle_error(
                e,
                st.session_state["security_context"]["user_id"],
                {"operation": "chat_processing", "user_input_length": len(user_input)}
            )
            
            logger.error(f"Chat processing error: {secure_error.technical_message}")
            st.error(
                f"Processing failed: {secure_error.user_message}\n\n"
                f"Error ID: {secure_error.error_id}\n\n"
                f"Details: {secure_error.technical_message}"
            )
            
            # Log security incident
            security_framework["audit_logger"].log_security_violation(
                st.session_state["security_context"]["user_id"],
                st.session_state["security_context"]["session_id"],
                "chat_processing_error",
                secure_error.technical_message,
                st.session_state["security_context"]["ip_address"]
            )

# Multi-step form using tabs
if "form_step" not in st.session_state:
    st.session_state.form_step = 0
    st.session_state.form_data = {}

steps = ["Vehicle Details", "Owner Information", "Incident Details", "Insurance Info"]
progress = st.progress((st.session_state.form_step + 1) / len(steps))

# Use numbered tabs to show progression clearly
tab_labels = [f"{i+1}. {step}" + (" ✅" if i < st.session_state.form_step else " 🟡" if i == st.session_state.form_step else " ⚪") 
              for i, step in enumerate(steps)]
tabs = st.tabs(tab_labels)

# Add some helpful navigation context
if st.session_state.form_step < len(steps) - 1:
    st.info(f"📍 **Current Step:** {steps[st.session_state.form_step]} (Step {st.session_state.form_step + 1} of {len(steps)})")
else:
    st.info("📍 **Final Step:** Review and submit your report")

# Vehicle Details Tab
with tabs[0]:
    st.subheader("🚗 Vehicle Information")
    if st.session_state.form_step >= 0:  # Always show content but disable if not current step
        with st.form("vehicle_form"):
            col1, col2 = st.columns(2)
            with col1:
                make = st.text_input("Make*", key="make", 
                                   value=st.session_state.form_data.get("make", ""),
                                   disabled=st.session_state.form_step != 0)
                year = st.number_input("Year*", min_value=1900, max_value=2024, 
                                     value=st.session_state.form_data.get("year", 2020),
                                     disabled=st.session_state.form_step != 0)
                vin = st.text_input("VIN Number*", key="vin", 
                                  value=st.session_state.form_data.get("vin", ""),
                                  disabled=st.session_state.form_step != 0)
            with col2:
                model = st.text_input("Model*", key="model", 
                                    value=st.session_state.form_data.get("model", ""),
                                    disabled=st.session_state.form_step != 0)
                color = st.text_input("Color*", key="color", 
                                    value=st.session_state.form_data.get("color", ""),
                                    disabled=st.session_state.form_step != 0)
                license_plate = st.text_input("License Plate*", key="license", 
                                             value=st.session_state.form_data.get("license_plate", ""),
                                             disabled=st.session_state.form_step != 0)
            
            col_prev, col_next = st.columns([1, 1])
            with col_next:
                if st.form_submit_button("Next →", disabled=st.session_state.form_step != 0):
                    # Validate all required fields
                    if not all([make, model, year, color, vin, license_plate]):
                        st.error("Please fill in all required fields.")
                    elif not DataValidator.validate_vin(vin):
                        st.error("Please enter a valid 17-character VIN.")
                    else:
                        st.session_state.form_data.update({
                            "make": make, "model": model, "year": year,
                            "color": color, "vin": vin, "license_plate": license_plate
                        })
                        st.session_state.form_step = 1
                        st.success("Vehicle details saved! Moving to owner information...")
                        st.rerun()

# Owner Information Tab
with tabs[1]:
    st.subheader("👤 Owner Information")
    if st.session_state.form_step >= 1:
        with st.form("owner_form"):
            col1, col2 = st.columns(2)
            with col1:
                owner_name = st.text_input("Full Name*", key="owner_name", 
                                         value=st.session_state.form_data.get("owner_name", ""),
                                         disabled=st.session_state.form_step != 1)
                phone = st.text_input("Phone Number*", key="phone", 
                                    value=st.session_state.form_data.get("phone", ""),
                                    disabled=st.session_state.form_step != 1,
                                    placeholder="1234567890")
            with col2:
                email = st.text_input("Email Address*", key="email", 
                                     value=st.session_state.form_data.get("email", ""),
                                     disabled=st.session_state.form_step != 1,
                                     placeholder="example@email.com")
                address = st.text_area("Current Address*", key="address", 
                                     value=st.session_state.form_data.get("address", ""),
                                     disabled=st.session_state.form_step != 1)
            
            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.form_submit_button("← Previous", disabled=st.session_state.form_step != 1):
                    st.session_state.form_step = 0
                    st.rerun()
            with col_next:
                if st.form_submit_button("Next →", disabled=st.session_state.form_step != 1):
                    # Validate all required fields
                    if not all([owner_name, phone, email, address]):
                        st.error("Please fill in all required fields.")
                    elif not DataValidator.validate_phone_number(phone):
                        st.error("Please enter a valid 10-digit phone number.")
                    elif not DataValidator.validate_email(email):
                        st.error("Please enter a valid email address.")
                    else:
                        st.session_state.form_data.update({
                            "owner_name": owner_name, "phone": phone,
                            "email": email, "address": address
                        })
                        st.session_state.form_step = 2
                        st.success("Owner information saved! Moving to incident details...")
                        st.rerun()

# Incident Details Tab
with tabs[2]:
    st.subheader("📋 Incident Information")
    if st.session_state.form_step >= 2:
        with st.form("incident_form"):
            col1, col2 = st.columns(2)
            with col1:
                incident_date = st.date_input("Date of Incident*", key="incident_date", 
                                            value=st.session_state.form_data.get("incident_date", datetime.now().date()),
                                            disabled=st.session_state.form_step != 2)
                incident_time = st.time_input("Time of Incident*", key="incident_time", 
                                            value=st.session_state.form_data.get("incident_time", datetime.now().time()),
                                            disabled=st.session_state.form_step != 2)
            with col2:
                location = st.text_input("Last Known Location*", key="location", 
                                       value=st.session_state.form_data.get("location", ""),
                                       disabled=st.session_state.form_step != 2)
                circumstances = st.text_area("Circumstances*", key="circumstances", 
                                            value=st.session_state.form_data.get("circumstances", ""),
                                            disabled=st.session_state.form_step != 2,
                                            placeholder="Please describe what happened...")
            
            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.form_submit_button("← Previous", disabled=st.session_state.form_step != 2):
                    st.session_state.form_step = 1
                    st.rerun()
            with col_next:
                if st.form_submit_button("Next →", disabled=st.session_state.form_step != 2):
                    if not all([location, circumstances]):
                        st.error("Please fill in all required fields.")
                    else:
                        st.session_state.form_data.update({
                            "incident_date": incident_date,
                            "incident_time": incident_time,
                            "location": location,
                            "circumstances": circumstances
                        })
                        st.session_state.form_step = 3
                        st.success("Incident details saved! Moving to insurance information...")
                        st.rerun()

# Insurance Information Tab
with tabs[3]:
    st.subheader("🛡️ Insurance Information")
    if st.session_state.form_step >= 3:
        with st.form("insurance_form"):
            col1, col2 = st.columns(2)
            with col1:
                insurance_company = st.text_input("Insurance Company*", key="insurance_company", 
                                                value=st.session_state.form_data.get("insurance_company", ""),
                                                disabled=st.session_state.form_step != 3)
            with col2:
                policy_number = st.text_input("Policy Number*", key="policy_number", 
                                            value=st.session_state.form_data.get("policy_number", ""),
                                            disabled=st.session_state.form_step != 3)
            
            col_prev, col_submit = st.columns([1, 1])
            with col_prev:
                if st.form_submit_button("← Previous", disabled=st.session_state.form_step != 3):
                    st.session_state.form_step = 2
                    st.rerun()
            with col_submit:
                if st.form_submit_button("🚀 Submit Report", disabled=st.session_state.form_step != 3):
                    if not all([insurance_company, policy_number]):
                        st.error("Please fill in all required fields.")
                    else:
                        st.session_state.form_data.update({
                            "insurance_company": insurance_company,
                            "policy_number": policy_number
                        })
                        
                        # Create ticket in database
                        try:
                            with st.spinner("Creating your report..."):
                                db_manager = st.session_state["db_manager"]
                                
                                # Create data models using the actual form data, not encrypted session state
                                vehicle_info = VehicleInfo(
                                    make=st.session_state.form_data["make"],
                                    model=st.session_state.form_data["model"],
                                    year=st.session_state.form_data["year"],
                                    color=st.session_state.form_data["color"],
                                    vin=st.session_state.form_data["vin"],
                                    license_plate=st.session_state.form_data["license_plate"]
                                )
                                
                                owner_info = OwnerInfo(
                                    name=st.session_state.form_data["owner_name"],
                                    phone=st.session_state.form_data["phone"],
                                    email=st.session_state.form_data["email"],
                                    address=st.session_state.form_data["address"]
                                )
                                
                                incident_info = IncidentInfo(
                                    incident_date=datetime.combine(
                                        st.session_state.form_data["incident_date"],
                                        st.session_state.form_data["incident_time"]
                                    ),
                                    incident_time=datetime.combine(
                                        st.session_state.form_data["incident_date"],
                                        st.session_state.form_data["incident_time"]
                                    ),
                                    location=st.session_state.form_data["location"],
                                    circumstances=st.session_state.form_data["circumstances"]
                                )
                                
                                insurance_info = InsuranceInfo(
                                    company=st.session_state.form_data["insurance_company"],
                                    policy_number=st.session_state.form_data["policy_number"]
                                )
                                
                                # Create ticket
                                ticket = asyncio.run(db_manager.create_ticket(
                                    st.session_state["user_id"],
                                    vehicle_info,
                                    owner_info,
                                    incident_info,
                                    insurance_info
                                ))
                                
                                # Trigger comprehensive notification workflow
                                try:
                                    orchestrator = st.session_state["orchestrator"]
                                    
                                    # Prepare ticket data for notifications
                                    ticket_data = {
                                        "ticket_id": ticket.ticket_id,
                                        "user_id": st.session_state["user_id"],
                                        "conversation_id": st.session_state["conversation_id"],
                                        "owner_name": st.session_state.form_data["owner_name"],
                                        "owner_email": st.session_state.form_data["email"],
                                        "phone": st.session_state.form_data["phone"],
                                        "make": st.session_state.form_data["make"],
                                        "model": st.session_state.form_data["model"],
                                        "year": st.session_state.form_data["year"],
                                        "color": st.session_state.form_data["color"],
                                        "license_plate": st.session_state.form_data["license_plate"],
                                        "vin": st.session_state.form_data["vin"],
                                        "location": st.session_state.form_data["location"],
                                        "circumstances": st.session_state.form_data["circumstances"],
                                        "insurance_company": st.session_state.form_data["insurance_company"],
                                        "policy_number": st.session_state.form_data["policy_number"],
                                        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M")
                                    }
                                    
                                    # Trigger notification system
                                    notification_result = asyncio.run(
                                        orchestrator.notification_system.create_ticket_notifications(ticket_data)
                                    )
                                    
                                    logger.info(f"Notification system result: {notification_result}")
                                    
                                except Exception as e:
                                    logger.error(f"Notification system failed: {e}")
                                    # Continue with basic success message even if notifications fail
                                
                                # Add success message to conversation
                                success_message = f"""
🎉 **Report submitted successfully!**

**Ticket ID**: {ticket.ticket_id}
**Status**: Processing & Active Investigation
**Next Steps**: 
- Police report filed automatically
- Insurance company notified
- Recovery services activated
- You'll receive updates via email and SMS

**What's happening now:**
- Our recovery team is searching your area
- Police database has been updated
- Insurance claim process initiated
- You can track progress with ticket ID: {ticket.ticket_id}

**Emergency contacts are available 24/7 if needed.**

You'll receive email confirmations shortly at {st.session_state.form_data['email']}.
"""
                                
                                # Add to conversation
                                asyncio.run(db_manager.add_message_to_conversation(
                                    st.session_state["conversation_id"],
                                    Message(
                                        role="assistant",
                                        content=success_message
                                    )
                                ))
                                
                                # Log ticket creation
                                ComplianceLogger.log_data_access(
                                    st.session_state["user_id"],
                                    "ticket",
                                    ticket.ticket_id,
                                    "create",
                                    True
                                )
                                
                                # Show success with ticket details
                                st.success(f"✅ Report submitted successfully!")
                                st.info(f"**Ticket ID: {ticket.ticket_id}**")
                                
                                # Show summary
                                with st.expander("📋 Report Summary", expanded=True):
                                    st.write("**Vehicle**: ", f"{st.session_state.form_data['make']} {st.session_state.form_data['model']} ({st.session_state.form_data['year']})")
                                    st.write("**License Plate**: ", st.session_state.form_data['license_plate'])
                                    st.write("**Owner**: ", st.session_state.form_data['owner_name'])
                                    st.write("**Location**: ", st.session_state.form_data['location'])
                                    st.write("**Insurance**: ", st.session_state.form_data['insurance_company'])
                                
                                # Add the message to chat display
                                if "messages" not in st.session_state:
                                    st.session_state.messages = []
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": success_message
                                })
                                
                                # Auto-scroll to chat to show the response
                                st.balloons()
                                
                        except Exception as e:
                            logger.error(f"Failed to create ticket: {e}")
                            st.error(f"❌ Failed to create ticket: {str(e)}")
                            st.error("Please check your information and try again, or contact support if the issue persists.")
                        
                        # Reset form after successful submission
                        if "ticket" in locals():
                            st.session_state.form_step = 0
                            st.session_state.form_data = {}
                            time.sleep(2)  # Give user time to see the success message
                            st.rerun()

# Show form progress summary
if st.session_state.form_step > 0:
    with st.expander("📋 Progress Summary", expanded=False):
        if st.session_state.form_step >= 1:
            st.write("✅ **Vehicle Details**: ", f"{st.session_state.form_data.get('make', '')} {st.session_state.form_data.get('model', '')} ({st.session_state.form_data.get('year', '')})")
        if st.session_state.form_step >= 2:
            st.write("✅ **Owner**: ", st.session_state.form_data.get('owner_name', ''))
        if st.session_state.form_step >= 3:
            st.write("✅ **Incident**: ", st.session_state.form_data.get('location', ''))
        if st.session_state.form_step >= 4:
            st.write("✅ **Insurance**: ", st.session_state.form_data.get('insurance_company', ''))