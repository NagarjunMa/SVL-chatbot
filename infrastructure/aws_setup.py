#!/usr/bin/env python3
"""
AWS Services Setup for SVL Chatbot - Bedrock Knowledge Base Migration
Sets up S3, OpenSearch, Bedrock Knowledge Base, and Guardrails
"""

import boto3
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional

class AWSServicesSetup:
    """Setup AWS services for Bedrock Knowledge Base implementation"""
    
    def __init__(self, region: str = "us-east-1", project_name: str = "svl-chatbot"):
        self.region = region
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d")
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=region)
        self.opensearch_client = boto3.client('opensearch', region_name=region)
        self.bedrock_client = boto3.client('bedrock', region_name=region)
        self.bedrock_agent_client = boto3.client('bedrock-agent', region_name=region)
        self.iam_client = boto3.client('iam', region_name=region)
        
        # Resource names
        self.s3_bucket_name = f"{project_name}-knowledge-base-{self.timestamp}"
        self.opensearch_domain_name = f"{project_name}-vector-search"
        self.knowledge_base_name = f"{project_name}-kb"
        self.guardrails_name = f"{project_name}-guardrails"
        
        print(f"🚀 Initializing AWS Services Setup for {project_name}")
        print(f"📍 Region: {region}")
        print(f"🪣 S3 Bucket: {self.s3_bucket_name}")
        print(f"🔍 OpenSearch Domain: {self.opensearch_domain_name}")
    
    def setup_s3_bucket(self) -> Dict[str, Any]:
        """Create S3 bucket for knowledge base documents"""
        try:
            print(f"\n📁 Creating S3 bucket: {self.s3_bucket_name}")
            
            # Create bucket
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.s3_bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.s3_bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=self.s3_bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Set bucket policy for Bedrock access
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AllowBedrockKnowledgeBaseAccess",
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "bedrock.amazonaws.com"
                        },
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{self.s3_bucket_name}",
                            f"arn:aws:s3:::{self.s3_bucket_name}/*"
                        ]
                    }
                ]
            }
            
            self.s3_client.put_bucket_policy(
                Bucket=self.s3_bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            print(f"✅ S3 bucket created successfully: {self.s3_bucket_name}")
            return {
                "status": "success",
                "bucket_name": self.s3_bucket_name,
                "bucket_arn": f"arn:aws:s3:::{self.s3_bucket_name}"
            }
            
        except Exception as e:
            print(f"❌ Failed to create S3 bucket: {e}")
            return {"status": "error", "error": str(e)}
    
    def setup_opensearch_domain(self) -> Dict[str, Any]:
        """Create OpenSearch domain for vector storage"""
        try:
            print(f"\n🔍 Creating OpenSearch domain: {self.opensearch_domain_name}")
            
            # OpenSearch domain configuration
            domain_config = {
                'DomainName': self.opensearch_domain_name,
                'EngineVersion': 'OpenSearch_2.9',
                'ClusterConfig': {
                    'InstanceType': 't3.small.search',  # Cost-effective for learning
                    'InstanceCount': 1,
                    'DedicatedMasterEnabled': False
                },
                'EBSOptions': {
                    'EBSEnabled': True,
                    'VolumeType': 'gp3',
                    'VolumeSize': 20  # GB
                },
                'AccessPolicies': json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "bedrock.amazonaws.com"
                            },
                            "Action": "es:*",
                            "Resource": f"arn:aws:es:{self.region}:*:domain/{self.opensearch_domain_name}/*"
                        }
                    ]
                }),
                'DomainEndpointOptions': {
                    'EnforceHTTPS': True
                },
                'NodeToNodeEncryptionOptions': {
                    'Enabled': True
                },
                'EncryptionAtRestOptions': {
                    'Enabled': True
                },
                'AdvancedSecurityOptions': {
                    'Enabled': False  # Simplified for learning
                }
            }
            
            response = self.opensearch_client.create_domain(**domain_config)
            
            print(f"⏳ OpenSearch domain creation initiated...")
            print(f"📝 Domain ARN: {response['DomainStatus']['ARN']}")
            print(f"⚠️  Note: Domain creation takes 10-15 minutes. Check AWS Console for progress.")
            
            return {
                "status": "creating",
                "domain_name": self.opensearch_domain_name,
                "domain_arn": response['DomainStatus']['ARN']
            }
            
        except Exception as e:
            print(f"❌ Failed to create OpenSearch domain: {e}")
            return {"status": "error", "error": str(e)}
    
    def setup_iam_roles(self) -> Dict[str, Any]:
        """Create IAM roles for Bedrock Knowledge Base"""
        try:
            print(f"\n🔐 Creating IAM roles for Bedrock Knowledge Base")
            
            # Knowledge Base service role
            kb_role_name = f"{self.project_name}-kb-role"
            
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "bedrock.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            # Create role
            role_response = self.iam_client.create_role(
                RoleName=kb_role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"Service role for {self.project_name} Bedrock Knowledge Base"
            )
            
            # Attach policies
            kb_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{self.s3_bucket_name}",
                            f"arn:aws:s3:::{self.s3_bucket_name}/*"
                        ]
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "es:ESHttpPost",
                            "es:ESHttpPut",
                            "es:ESHttpGet",
                            "es:ESHttpDelete"
                        ],
                        "Resource": f"arn:aws:es:{self.region}:*:domain/{self.opensearch_domain_name}/*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:InvokeModel"
                        ],
                        "Resource": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1"
                    }
                ]
            }
            
            self.iam_client.put_role_policy(
                RoleName=kb_role_name,
                PolicyName=f"{self.project_name}-kb-policy",
                PolicyDocument=json.dumps(kb_policy)
            )
            
            print(f"✅ IAM role created: {kb_role_name}")
            return {
                "status": "success",
                "role_name": kb_role_name,
                "role_arn": role_response['Role']['Arn']
            }
            
        except Exception as e:
            print(f"❌ Failed to create IAM roles: {e}")
            return {"status": "error", "error": str(e)}
    
    def upload_pdfs_to_s3(self, local_data_path: str = "./data") -> Dict[str, Any]:
        """Upload existing PDFs to S3 bucket"""
        try:
            print(f"\n📤 Uploading PDFs from {local_data_path} to S3...")
            
            uploaded_files = []
            
            # Walk through data directory
            for root, dirs, files in os.walk(local_data_path):
                for file in files:
                    if file.endswith('.pdf'):
                        local_path = os.path.join(root, file)
                        
                        # Create S3 key maintaining directory structure
                        relative_path = os.path.relpath(local_path, local_data_path)
                        s3_key = f"documents/{relative_path}"
                        
                        # Upload file
                        self.s3_client.upload_file(
                            local_path, 
                            self.s3_bucket_name, 
                            s3_key,
                            ExtraArgs={
                                'ContentType': 'application/pdf',
                                'Metadata': {
                                    'source': 'svl-migration',
                                    'upload_date': datetime.now().isoformat()
                                }
                            }
                        )
                        
                        uploaded_files.append({
                            "local_path": local_path,
                            "s3_key": s3_key,
                            "s3_url": f"s3://{self.s3_bucket_name}/{s3_key}"
                        })
                        
                        print(f"📄 Uploaded: {relative_path}")
            
            print(f"✅ Successfully uploaded {len(uploaded_files)} PDF files")
            return {
                "status": "success",
                "uploaded_files": uploaded_files,
                "bucket_name": self.s3_bucket_name
            }
            
        except Exception as e:
            print(f"❌ Failed to upload PDFs: {e}")
            return {"status": "error", "error": str(e)}
    
    def create_bedrock_knowledge_base(self, opensearch_endpoint: str, role_arn: str) -> Dict[str, Any]:
        """Create Bedrock Knowledge Base"""
        try:
            print(f"\n🧠 Creating Bedrock Knowledge Base: {self.knowledge_base_name}")
            
            kb_config = {
                'name': self.knowledge_base_name,
                'description': f'SVL Chatbot Knowledge Base - migrated from FAISS on {datetime.now().strftime("%Y-%m-%d")}',
                'roleArn': role_arn,
                'knowledgeBaseConfiguration': {
                    'type': 'VECTOR',
                    'vectorKnowledgeBaseConfiguration': {
                        'embeddingModelArn': f'arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v1'
                    }
                },
                'storageConfiguration': {
                    'type': 'OPENSEARCH_SERVERLESS',
                    'opensearchServerlessConfiguration': {
                        'collectionArn': opensearch_endpoint,
                        'vectorIndexName': 'svl-vector-index',
                        'fieldMapping': {
                            'vectorField': 'vector',
                            'textField': 'text',
                            'metadataField': 'metadata'
                        }
                    }
                }
            }
            
            response = self.bedrock_agent_client.create_knowledge_base(**kb_config)
            
            print(f"✅ Knowledge Base created successfully!")
            print(f"📝 Knowledge Base ID: {response['knowledgeBase']['knowledgeBaseId']}")
            
            return {
                "status": "success",
                "knowledge_base_id": response['knowledgeBase']['knowledgeBaseId'],
                "knowledge_base_arn": response['knowledgeBase']['knowledgeBaseArn']
            }
            
        except Exception as e:
            print(f"❌ Failed to create Knowledge Base: {e}")
            return {"status": "error", "error": str(e)}
    
    def create_guardrails(self) -> Dict[str, Any]:
        """Create Bedrock Guardrails"""
        try:
            print(f"\n🛡️ Creating Bedrock Guardrails: {self.guardrails_name}")
            
            guardrails_config = {
                'name': self.guardrails_name,
                'description': 'SVL Chatbot content filtering and safety guardrails',
                'topicPolicyConfig': {
                    'topicsConfig': [
                        {
                            'name': 'Violence',
                            'definition': 'Content related to violence, threats, or harmful activities',
                            'examples': ['I will hurt someone', 'Violence is the answer'],
                            'type': 'DENY'
                        },
                        {
                            'name': 'Illegal Activities',
                            'definition': 'Content promoting illegal activities or law breaking',
                            'examples': ['How to break into cars', 'Illegal activities'],
                            'type': 'DENY'
                        }
                    ]
                },
                'contentPolicyConfig': {
                    'filtersConfig': [
                        {
                            'type': 'SEXUAL',
                            'inputStrength': 'HIGH',
                            'outputStrength': 'HIGH'
                        },
                        {
                            'type': 'VIOLENCE',
                            'inputStrength': 'HIGH',
                            'outputStrength': 'HIGH'
                        },
                        {
                            'type': 'HATE',
                            'inputStrength': 'HIGH',
                            'outputStrength': 'HIGH'
                        },
                        {
                            'type': 'INSULTS',
                            'inputStrength': 'MEDIUM',
                            'outputStrength': 'MEDIUM'
                        }
                    ]
                },
                'wordPolicyConfig': {
                    'wordsConfig': [
                        {
                            'text': 'damn'
                        },
                        {
                            'text': 'hell'
                        }
                    ],
                    'managedWordListsConfig': [
                        {
                            'type': 'PROFANITY'
                        }
                    ]
                },
                'sensitiveInformationPolicyConfig': {
                    'piiEntitiesConfig': [
                        {
                            'type': 'EMAIL',
                            'action': 'BLOCK'
                        },
                        {
                            'type': 'PHONE',
                            'action': 'BLOCK'
                        },
                        {
                            'type': 'SSN',
                            'action': 'BLOCK'
                        },
                        {
                            'type': 'CREDIT_DEBIT_CARD_NUMBER',
                            'action': 'BLOCK'
                        }
                    ]
                }
            }
            
            response = self.bedrock_client.create_guardrail(**guardrails_config)
            
            print(f"✅ Guardrails created successfully!")
            print(f"📝 Guardrail ID: {response['guardrailId']}")
            
            return {
                "status": "success",
                "guardrail_id": response['guardrailId'],
                "guardrail_arn": response['guardrailArn']
            }
            
        except Exception as e:
            print(f"❌ Failed to create Guardrails: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_setup_summary(self) -> Dict[str, Any]:
        """Get summary of all created resources"""
        return {
            "s3_bucket": self.s3_bucket_name,
            "opensearch_domain": self.opensearch_domain_name,
            "knowledge_base_name": self.knowledge_base_name,
            "guardrails_name": self.guardrails_name,
            "region": self.region,
            "project": self.project_name
        }

def main():
    """Main setup function"""
    print("🚀 SVL Chatbot - AWS Services Setup")
    print("📋 This will create: S3 Bucket, OpenSearch Domain, Bedrock Knowledge Base, Guardrails")
    
    # Initialize setup
    setup = AWSServicesSetup()
    
    # Step 1: Create S3 bucket
    s3_result = setup.setup_s3_bucket()
    if s3_result["status"] != "success":
        print("❌ S3 setup failed. Exiting.")
        return
    
    # Step 2: Upload PDFs
    upload_result = setup.upload_pdfs_to_s3()
    if upload_result["status"] != "success":
        print("❌ PDF upload failed. Exiting.")
        return
    
    # Step 3: Create IAM roles
    iam_result = setup.setup_iam_roles()
    if iam_result["status"] != "success":
        print("❌ IAM setup failed. Exiting.")
        return
    
    # Step 4: Create OpenSearch domain
    opensearch_result = setup.setup_opensearch_domain()
    if opensearch_result["status"] == "error":
        print("❌ OpenSearch setup failed. Exiting.")
        return
    
    # Step 5: Create Guardrails
    guardrails_result = setup.create_guardrails()
    if guardrails_result["status"] != "success":
        print("❌ Guardrails setup failed.")
    
    # Summary
    print("\n" + "="*60)
    print("🎉 AWS SERVICES SETUP SUMMARY")
    print("="*60)
    print(f"✅ S3 Bucket: {setup.s3_bucket_name}")
    print(f"✅ OpenSearch Domain: {setup.opensearch_domain_name} (creating...)")
    print(f"✅ IAM Role: {iam_result['role_name']}")
    print(f"✅ Guardrails: {setup.guardrails_name}")
    print(f"📤 PDFs Uploaded: {len(upload_result['uploaded_files'])} files")
    print("\n⏳ Next Steps:")
    print("1. Wait for OpenSearch domain to be active (10-15 minutes)")
    print("2. Run knowledge base creation after OpenSearch is ready")
    print("3. Update application configuration with new resource IDs")
    print("\n🔗 Check AWS Console:")
    print(f"   S3: https://console.aws.amazon.com/s3/buckets/{setup.s3_bucket_name}")
    print(f"   OpenSearch: https://console.aws.amazon.com/es/home?region={setup.region}")
    print(f"   Bedrock: https://console.aws.amazon.com/bedrock/home?region={setup.region}")

if __name__ == "__main__":
    main() 