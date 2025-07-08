#!/usr/bin/env python3
"""
Update IAM role permissions to include OpenSearch access
"""

import boto3
import json

def update_iam_role_policy():
    """Update the IAM role with OpenSearch permissions"""
    iam_client = boto3.client('iam', region_name='us-east-1')
    
    try:
        print("🔧 Updating IAM role permissions for OpenSearch access...")
        
        # Define the updated policy document
        updated_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:Retrieve",
                        "bedrock:RetrieveAndGenerate"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        "arn:aws:s3:::svl-chatbot-knowledge-base-20250708",
                        "arn:aws:s3:::svl-chatbot-knowledge-base-20250708/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "es:DescribeDomain",
                        "es:DescribeElasticsearchDomain",
                        "es:ESHttpGet",
                        "es:ESHttpPost",
                        "es:ESHttpPut",
                        "es:ESHttpDelete",
                        "es:ESHttpHead",
                        "es:DescribeDomains",
                        "es:DescribeElasticsearchDomains"
                    ],
                    "Resource": [
                        "arn:aws:es:us-east-1:790744020566:domain/svl-chatbot-vector-search",
                        "arn:aws:es:us-east-1:790744020566:domain/svl-chatbot-vector-search/*"
                    ]
                }
            ]
        }
        
        # Update the policy
        response = iam_client.put_role_policy(
            RoleName='svl-chatbot-kb-role',
            PolicyName='SVLChatbotKnowledgeBasePolicy',
            PolicyDocument=json.dumps(updated_policy)
        )
        
        print("✅ IAM role policy updated successfully!")
        print("📝 Policy includes permissions for:")
        print("   - Bedrock model invocation")
        print("   - S3 bucket access")
        print("   - OpenSearch domain operations")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"❌ Failed to update IAM role policy: {e}")
        return {"status": "error", "error": str(e)}

def update_opensearch_access_policy():
    """Update OpenSearch domain access policy to allow our IAM role"""
    opensearch_client = boto3.client('opensearch', region_name='us-east-1')
    
    try:
        print("\n🔧 Updating OpenSearch domain access policy...")
        
        # Define access policy for the domain
        access_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": [
                            "arn:aws:iam::790744020566:role/svl-chatbot-kb-role",
                            "arn:aws:iam::790744020566:root"
                        ]
                    },
                    "Action": "es:*",
                    "Resource": "arn:aws:es:us-east-1:790744020566:domain/svl-chatbot-vector-search/*"
                }
            ]
        }
        
        # Update the domain access policy
        response = opensearch_client.update_domain_config(
            DomainName='svl-chatbot-vector-search',
            AccessPolicies=json.dumps(access_policy)
        )
        
        print("✅ OpenSearch domain access policy updated!")
        print("📝 IAM role now has access to the OpenSearch domain")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"❌ Failed to update OpenSearch access policy: {e}")
        return {"status": "error", "error": str(e)}

def main():
    """Main function to update permissions"""
    print("🔐 Updating IAM and OpenSearch Permissions...")
    
    # Step 1: Update IAM role policy
    iam_result = update_iam_role_policy()
    if iam_result["status"] != "success":
        print("❌ Failed to update IAM role policy")
        return
    
    # Step 2: Update OpenSearch access policy
    opensearch_result = update_opensearch_access_policy()
    if opensearch_result["status"] != "success":
        print("❌ Failed to update OpenSearch access policy")
        return
    
    print("\n" + "="*60)
    print("🎉 PERMISSIONS UPDATED SUCCESSFULLY!")
    print("="*60)
    print("✅ IAM role has Bedrock, S3, and OpenSearch permissions")
    print("✅ OpenSearch domain allows access from IAM role")
    print("\n⏳ Wait 1-2 minutes for policies to propagate, then retry Knowledge Base creation!")

if __name__ == "__main__":
    main() 