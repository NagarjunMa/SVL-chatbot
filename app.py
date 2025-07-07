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
from data.models import Message, VehicleInfo, OwnerInfo, IncidentInfo, InsuranceInfo, Ticket

# Load environment variables
load_dotenv()

# Set up logger
logger = get_logger()

# Custom CSS for styling
with open("./utils/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Stolen Vehicle Locator Chatbot", page_icon="🚗", layout="centered")
st.title("🚗 Stolen Vehicle Locator (SVL) Chatbot")

# Initialize session state
initialize_session()

# Initialize database manager
if "db_manager" not in st.session_state:
    try:
        st.session_state["db_manager"] = DatabaseManager()
        logger.info("Database manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database manager: {e}")
        st.error("Critical error: Unable to connect to database.")
        st.stop()

# Initialize ConversationManager in session state
if "conversation_manager" not in st.session_state:
    try:
        st.session_state["conversation_manager"] = get_conversation_manager(
            session_id=st.session_state.get("conversation_id", "default-session")
        )
    except Exception as e:
        logger.error(f"Failed to initialize ConversationManager: {e}")
        st.error("Critical error: Unable to connect to AI service.")
        st.stop()

# Initialize user session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = f"user-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

# Initialize conversation ID
if "conversation_id" not in st.session_state:
    try:
        # Get or create conversation for user
        db_manager = st.session_state["db_manager"]
        conversation_id = asyncio.run(db_manager.get_or_create_session(st.session_state["user_id"]))
        st.session_state["conversation_id"] = conversation_id
        logger.info(f"Initialized conversation: {conversation_id}")
    except Exception as e:
        logger.error(f"Failed to initialize conversation: {e}")
        st.session_state["conversation_id"] = "fallback-conversation"

# Helper: Determine conversation phase (simple heuristic, can be improved)
def get_phase():
    messages = st.session_state["messages"]
    if len(messages) <= 1:
        return "greeting"
    # Example: Use form step if available
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
    # Fallback: FAQ or process explanation
    last_user = [m for m in messages[::-1] if m["role"] == "user"]
    if last_user and any(q in last_user[0]["content"].lower() for q in ["how", "what", "when", "where", "faq"]):
        return "faq"
    return "process_explanation"

# Chat interface
with st.container():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("How can I help you locate a stolen vehicle today?")
    if user_input:
        try:
            # Validate and sanitize user input
            sanitized_input = DataValidator.sanitize_text(user_input)
            if not sanitized_input:
                st.error("Please enter a valid message.")
            else:
                # Check for PII in user input
                pii_detected = PIIHandler.detect_pii(sanitized_input)
                if pii_detected:
                    logger.warning(f"PII detected in user input: {pii_detected}")
                    ComplianceLogger.log_pii_access(
                        st.session_state["user_id"], 
                        list(pii_detected.keys())[0], 
                        st.session_state["conversation_id"], 
                        "user_input", 
                        False
                    )
                    st.warning("For your privacy, please avoid sharing personal information in the chat.")
                
                # Add user message to session state
                append_message("user", sanitized_input)
                
                # Process the response
                phase = get_phase()
                logger.info(f"Processing user input in phase: {phase}")
                
                # Get AI response
                cm = st.session_state["conversation_manager"]
                logger.info("About to call conversation manager...")
                response = cm.process_user_input(sanitized_input, phase)
                logger.info(f"Received response from conversation manager: {response[:100]}...")
                
                # Add assistant response
                append_message("assistant", response)
                logger.info(f"Assistant response added to messages")
                
                # Save conversation to database (in background)
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
                    
                    # Log data access
                    ComplianceLogger.log_data_access(
                        st.session_state["user_id"],
                        "conversation",
                        st.session_state["conversation_id"],
                        "add_message",
                        True
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to save conversation to database: {e}")
                    # Don't show error to user, just log it
                
                # Refresh the page to show new messages
                st.rerun()
                
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            st.error("Sorry, something went wrong. Please try again later.")

# Sidebar for emergency contacts and information
with st.sidebar:
    st.header("🚨 Emergency Contacts")
    st.info("""
    **Police Emergency**: 911
    **SVL Hotline**: 1-800-555-0123
    **Insurance Support**: 1-800-555-0199
    """)
    st.markdown("---")
    st.caption("This system is monitored 24/7. For immediate assistance, please call emergency services.")

# Multi-step form using tabs
if "form_step" not in st.session_state:
    st.session_state.form_step = 0
    st.session_state.form_data = {}

steps = ["Vehicle Details", "Owner Information", "Incident Details", "Insurance Info"]
progress = st.progress((st.session_state.form_step + 1) / len(steps))
tabs = st.tabs(steps)

with tabs[0]:  # Vehicle Details
    if st.session_state.form_step == 0:
        with st.form("vehicle_form"):
            col1, col2 = st.columns(2)
            with col1:
                make = st.text_input("Make*", key="make")
                year = st.number_input("Year*", min_value=1900, max_value=2024, value=2020)
                vin = st.text_input("VIN Number*", key="vin")
            with col2:
                model = st.text_input("Model*", key="model")
                color = st.text_input("Color*", key="color")
                license_plate = st.text_input("License Plate*", key="license")
            
            if st.form_submit_button("Next"):
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
                    st.rerun()

with tabs[1]:  # Owner Information
    if st.session_state.form_step == 1:
        with st.form("owner_form"):
            col1, col2 = st.columns(2)
            with col1:
                owner_name = st.text_input("Full Name*", key="owner_name")
                phone = st.text_input("Phone Number*", key="phone")
            with col2:
                email = st.text_input("Email Address*", key="email")
                address = st.text_area("Current Address*", key="address")
            
            col3, col4 = st.columns([1, 1])
            with col3:
                if st.form_submit_button("Previous"):
                    st.session_state.form_step = 0
                    st.rerun()
            with col4:
                if st.form_submit_button("Next"):
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
                        st.rerun()

with tabs[2]:  # Incident Details
    if st.session_state.form_step == 2:
        with st.form("incident_form"):
            col1, col2 = st.columns(2)
            with col1:
                incident_date = st.date_input("Date of Incident*", key="incident_date")
                incident_time = st.time_input("Time of Incident*", key="incident_time")
            with col2:
                location = st.text_input("Last Known Location*", key="location")
                circumstances = st.text_area("Circumstances*", key="circumstances")
            
            col3, col4 = st.columns([1, 1])
            with col3:
                if st.form_submit_button("Previous"):
                    st.session_state.form_step = 1
                    st.rerun()
            with col4:
                if st.form_submit_button("Next"):
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
                        st.rerun()

with tabs[3]:  # Insurance Information
    if st.session_state.form_step == 3:
        with st.form("insurance_form"):
            col1, col2 = st.columns(2)
            with col1:
                insurance_company = st.text_input("Insurance Company*", key="insurance_company")
            with col2:
                policy_number = st.text_input("Policy Number*", key="policy_number")
            
            col3, col4 = st.columns([1, 1])
            with col3:
                if st.form_submit_button("Previous"):
                    st.session_state.form_step = 2
                    st.rerun()
            with col4:
                if st.form_submit_button("Submit Report"):
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
                                
                                # Create data models
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
                                
                                # Update conversation with ticket ID
                                asyncio.run(db_manager.add_message_to_conversation(
                                    st.session_state["conversation_id"],
                                    Message(
                                        role="system",
                                        content=f"Ticket created: {ticket.ticket_id}"
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
                                
                                st.success(f"Report submitted successfully! Ticket ID: {ticket.ticket_id}")
                                st.json(st.session_state.form_data)
                                
                        except Exception as e:
                            logger.error(f"Failed to create ticket: {e}")
                            st.error("Failed to create ticket. Please try again.")
                        
                        # Reset form
                        st.session_state.form_step = 0
                        st.session_state.form_data = {}
                        st.rerun()