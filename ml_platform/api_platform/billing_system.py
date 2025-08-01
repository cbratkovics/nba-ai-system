"""
Subscription and Billing System with Stripe Integration
Usage-based billing, quota management, and customer analytics
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import stripe
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Gauge, Histogram
import logging
from decimal import Decimal
import hmac
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Metrics
billing_events = Counter('billing_events_total', 'Billing events', ['event_type', 'customer'])
usage_metrics = Gauge('customer_usage', 'Customer usage metrics', ['customer_id', 'metric_type'])
revenue_metrics = Gauge('revenue_metrics', 'Revenue metrics', ['period', 'tier'])
subscription_count = Gauge('active_subscriptions', 'Active subscriptions', ['tier'])

Base = declarative_base()
logger = logging.getLogger(__name__)


class SubscriptionTier(Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class BillingPeriod(Enum):
    MONTHLY = "monthly"
    YEARLY = "yearly"


class UsageType(Enum):
    API_REQUEST = "api_request"
    PREDICTION = "prediction"
    BULK_PREDICTION = "bulk_prediction"
    STREAMING_HOUR = "streaming_hour"
    STORAGE_GB = "storage_gb"


@dataclass
class TierConfig:
    name: str
    base_price: Decimal
    included_requests: int
    overage_price: Decimal  # per request
    features: List[str]
    rate_limits: Dict[str, int]


# Database Models
class CustomerDB(Base):
    __tablename__ = 'customers'
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True)
    company_name = Column(String)
    stripe_customer_id = Column(String, unique=True)
    subscription_tier = Column(String)
    billing_period = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)


class SubscriptionDB(Base):
    __tablename__ = 'subscriptions'
    
    id = Column(String, primary_key=True)
    customer_id = Column(String)
    stripe_subscription_id = Column(String, unique=True)
    tier = Column(String)
    status = Column(String)  # active, canceled, past_due, etc.
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


class UsageRecordDB(Base):
    __tablename__ = 'usage_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String)
    usage_type = Column(String)
    quantity = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    metadata = Column(JSON)


class InvoiceDB(Base):
    __tablename__ = 'invoices'
    
    id = Column(String, primary_key=True)
    customer_id = Column(String)
    stripe_invoice_id = Column(String, unique=True)
    amount_due = Column(Integer)  # in cents
    amount_paid = Column(Integer)
    status = Column(String)
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    line_items = Column(JSON)


# Pydantic Models
class CustomerCreate(BaseModel):
    email: str
    company_name: Optional[str] = None
    subscription_tier: SubscriptionTier
    billing_period: BillingPeriod = BillingPeriod.MONTHLY


class UsageRecord(BaseModel):
    customer_id: str
    usage_type: UsageType
    quantity: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class BillingSystemConfig:
    """Configuration for different subscription tiers"""
    
    TIERS = {
        SubscriptionTier.STARTER: TierConfig(
            name="Starter",
            base_price=Decimal("99.00"),
            included_requests=10000,
            overage_price=Decimal("0.01"),
            features=["Basic API Access", "Email Support", "Documentation"],
            rate_limits={"requests_per_minute": 100, "predictions_per_day": 1000}
        ),
        SubscriptionTier.PROFESSIONAL: TierConfig(
            name="Professional",
            base_price=Decimal("499.00"),
            included_requests=100000,
            overage_price=Decimal("0.008"),
            features=[
                "Full API Access", "Real-time Predictions", "Priority Support",
                "Advanced Analytics", "Webhooks", "Custom Models"
            ],
            rate_limits={"requests_per_minute": 1000, "predictions_per_day": 25000}
        ),
        SubscriptionTier.ENTERPRISE: TierConfig(
            name="Enterprise",
            base_price=Decimal("2499.00"),
            included_requests=1000000,
            overage_price=Decimal("0.005"),
            features=[
                "Unlimited API Access", "Real-time Streaming", "24/7 Support",
                "Custom Integration", "SLA Guarantee", "Dedicated Account Manager",
                "White-label Options", "Custom Models", "Advanced Security"
            ],
            rate_limits={"requests_per_minute": 10000, "predictions_per_day": 500000}
        )
    }


class BillingSystem:
    """Production billing system with Stripe integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Stripe configuration
        stripe.api_key = config['stripe_secret_key']
        self.stripe_webhook_secret = config['stripe_webhook_secret']
        
        # Database setup
        self.engine = create_engine(config['postgres_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis for caching and rate limiting
        self.redis_client = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        
        # Background task executor
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def create_customer(self, customer_data: CustomerCreate) -> Dict[str, Any]:
        """Create new customer with Stripe integration"""
        try:
            # Create Stripe customer
            stripe_customer = stripe.Customer.create(
                email=customer_data.email,
                metadata={
                    'company_name': customer_data.company_name or '',
                    'subscription_tier': customer_data.subscription_tier.value
                }
            )
            
            # Create subscription
            tier_config = BillingSystemConfig.TIERS[customer_data.subscription_tier]
            
            # Create Stripe price if doesn't exist
            price_id = await self._get_or_create_price(
                customer_data.subscription_tier,
                customer_data.billing_period,
                tier_config.base_price
            )
            
            # Create Stripe subscription
            stripe_subscription = stripe.Subscription.create(
                customer=stripe_customer.id,
                items=[{'price': price_id}],
                metadata={
                    'tier': customer_data.subscription_tier.value,
                    'billing_period': customer_data.billing_period.value
                }
            )
            
            # Store in database
            customer_id = f"cust_{int(time.time())}"
            
            with self.Session() as session:
                # Create customer record
                customer = CustomerDB(
                    id=customer_id,
                    email=customer_data.email,
                    company_name=customer_data.company_name,
                    stripe_customer_id=stripe_customer.id,
                    subscription_tier=customer_data.subscription_tier.value,
                    billing_period=customer_data.billing_period.value,
                    metadata={
                        'onboarding_completed': False,
                        'trial_end': None
                    }
                )
                
                # Create subscription record
                subscription = SubscriptionDB(
                    id=f"sub_{int(time.time())}",
                    customer_id=customer_id,
                    stripe_subscription_id=stripe_subscription.id,
                    tier=customer_data.subscription_tier.value,
                    status=stripe_subscription.status,
                    current_period_start=datetime.fromtimestamp(stripe_subscription.current_period_start),
                    current_period_end=datetime.fromtimestamp(stripe_subscription.current_period_end)
                )
                
                session.add(customer)
                session.add(subscription)
                session.commit()
            
            # Update metrics
            subscription_count.labels(tier=customer_data.subscription_tier.value).inc()
            billing_events.labels(event_type='customer_created', customer=customer_id).inc()
            
            logger.info(f"Created customer {customer_id} with tier {customer_data.subscription_tier.value}")
            
            return {
                'customer_id': customer_id,
                'stripe_customer_id': stripe_customer.id,
                'subscription_id': subscription.id,
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Failed to create customer: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    async def record_usage(self, usage: UsageRecord) -> None:
        """Record usage for billing"""
        try:
            # Get customer subscription info
            customer_info = await self._get_customer_info(usage.customer_id)
            
            if not customer_info:
                logger.warning(f"Customer {usage.customer_id} not found")
                return
            
            # Store usage record
            with self.Session() as session:
                usage_record = UsageRecordDB(
                    customer_id=usage.customer_id,
                    usage_type=usage.usage_type.value,
                    quantity=usage.quantity,
                    timestamp=usage.timestamp,
                    period_start=customer_info['current_period_start'],
                    period_end=customer_info['current_period_end'],
                    metadata=usage.metadata or {}
                )
                session.add(usage_record)
                session.commit()
            
            # Update Redis counters for real-time tracking
            period_key = f"usage:{usage.customer_id}:{usage.usage_type.value}:{usage.timestamp.strftime('%Y-%m')}"
            
            self.redis_client.incrby(period_key, usage.quantity)
            self.redis_client.expire(period_key, 86400 * 32)  # 32 days
            
            # Update Prometheus metrics
            usage_metrics.labels(
                customer_id=usage.customer_id,
                metric_type=usage.usage_type.value
            ).inc(usage.quantity)
            
            # Check if overage reporting needed
            await self._check_overage_reporting(usage.customer_id, usage.usage_type)
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
    
    async def _check_overage_reporting(self, customer_id: str, usage_type: UsageType) -> None:
        """Check and report usage overages to Stripe"""
        try:
            customer_info = await self._get_customer_info(customer_id)
            tier_config = BillingSystemConfig.TIERS[SubscriptionTier(customer_info['tier'])]
            
            # Get current period usage
            current_usage = await self._get_period_usage(
                customer_id,
                usage_type,
                customer_info['current_period_start'],
                customer_info['current_period_end']
            )
            
            # Check if over included amount
            if usage_type == UsageType.API_REQUEST and current_usage > tier_config.included_requests:
                overage = current_usage - tier_config.included_requests
                
                # Report overage to Stripe for billing
                await self._report_stripe_usage(
                    customer_info['stripe_subscription_id'],
                    overage,
                    tier_config.overage_price
                )
                
        except Exception as e:
            logger.error(f"Error checking overage for {customer_id}: {e}")
    
    async def _report_stripe_usage(self, subscription_id: str, quantity: int, unit_price: Decimal) -> None:
        """Report usage to Stripe for metered billing"""
        try:
            # Get subscription
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Find metered price item
            metered_item = None
            for item in subscription['items']['data']:
                price = stripe.Price.retrieve(item['price']['id'])
                if price['billing_scheme'] == 'per_unit':
                    metered_item = item
                    break
            
            if metered_item:
                # Create usage record in Stripe
                stripe.SubscriptionItem.create_usage_record(
                    metered_item['id'],
                    quantity=quantity,
                    timestamp=int(time.time())
                )
                
                logger.info(f"Reported {quantity} usage units to Stripe for subscription {subscription_id}")
                
        except Exception as e:
            logger.error(f"Failed to report Stripe usage: {e}")
    
    async def get_customer_usage_summary(self, customer_id: str) -> Dict[str, Any]:
        """Get comprehensive usage summary for customer"""
        try:
            customer_info = await self._get_customer_info(customer_id)
            
            if not customer_info:
                raise HTTPException(status_code=404, detail="Customer not found")
            
            tier_config = BillingSystemConfig.TIERS[SubscriptionTier(customer_info['tier'])]
            
            # Get current period usage
            period_usage = {}
            for usage_type in UsageType:
                usage = await self._get_period_usage(
                    customer_id,
                    usage_type,
                    customer_info['current_period_start'],
                    customer_info['current_period_end']
                )
                period_usage[usage_type.value] = usage
            
            # Calculate costs
            api_usage = period_usage.get(UsageType.API_REQUEST.value, 0)
            included = tier_config.included_requests
            overage = max(0, api_usage - included)
            overage_cost = overage * tier_config.overage_price
            
            return {
                'customer_id': customer_id,
                'tier': customer_info['tier'],
                'billing_period': customer_info['billing_period'],
                'period_start': customer_info['current_period_start'],
                'period_end': customer_info['current_period_end'],
                'base_cost': float(tier_config.base_price),
                'usage': period_usage,
                'included_requests': included,
                'overage_requests': overage,
                'overage_cost': float(overage_cost),
                'total_estimated_cost': float(tier_config.base_price + overage_cost),
                'rate_limits': tier_config.rate_limits,
                'features': tier_config.features
            }
            
        except Exception as e:
            logger.error(f"Error getting usage summary for {customer_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get usage summary")
    
    async def handle_stripe_webhook(self, request: Request) -> JSONResponse:
        """Handle Stripe webhook events"""
        try:
            payload = await request.body()
            sig_header = request.headers.get('stripe-signature')
            
            # Verify webhook signature
            try:
                event = stripe.Webhook.construct_event(
                    payload, sig_header, self.stripe_webhook_secret
                )
            except ValueError:
                logger.error("Invalid payload in Stripe webhook")
                raise HTTPException(status_code=400, detail="Invalid payload")
            except stripe.error.SignatureVerificationError:
                logger.error("Invalid signature in Stripe webhook")
                raise HTTPException(status_code=400, detail="Invalid signature")
            
            # Handle different event types
            if event['type'] == 'customer.subscription.created':
                await self._handle_subscription_created(event['data']['object'])
            elif event['type'] == 'customer.subscription.updated':
                await self._handle_subscription_updated(event['data']['object'])
            elif event['type'] == 'customer.subscription.deleted':
                await self._handle_subscription_deleted(event['data']['object'])
            elif event['type'] == 'invoice.payment_succeeded':
                await self._handle_payment_succeeded(event['data']['object'])
            elif event['type'] == 'invoice.payment_failed':
                await self._handle_payment_failed(event['data']['object'])
            
            return JSONResponse(content={'status': 'success'})
            
        except Exception as e:
            logger.error(f"Error handling Stripe webhook: {e}")
            raise HTTPException(status_code=500, detail="Webhook processing failed")
    
    async def _handle_subscription_created(self, subscription: Dict[str, Any]) -> None:
        """Handle subscription creation event"""
        try:
            with self.Session() as session:
                # Update subscription record
                sub_record = session.query(SubscriptionDB).filter_by(
                    stripe_subscription_id=subscription['id']
                ).first()
                
                if sub_record:
                    sub_record.status = subscription['status']
                    sub_record.current_period_start = datetime.fromtimestamp(subscription['current_period_start'])
                    sub_record.current_period_end = datetime.fromtimestamp(subscription['current_period_end'])
                    session.commit()
            
            billing_events.labels(event_type='subscription_created', customer=subscription['customer']).inc()
            
        except Exception as e:
            logger.error(f"Error handling subscription created: {e}")
    
    async def _handle_payment_succeeded(self, invoice: Dict[str, Any]) -> None:
        """Handle successful payment"""
        try:
            with self.Session() as session:
                # Create or update invoice record
                invoice_record = InvoiceDB(
                    id=f"inv_{int(time.time())}",
                    stripe_invoice_id=invoice['id'],
                    customer_id=await self._get_customer_id_from_stripe_customer(invoice['customer']),
                    amount_due=invoice['amount_due'],
                    amount_paid=invoice['amount_paid'],
                    status=invoice['status'],
                    period_start=datetime.fromtimestamp(invoice['period_start']),
                    period_end=datetime.fromtimestamp(invoice['period_end']),
                    line_items=invoice.get('lines', {})
                )
                session.add(invoice_record)
                session.commit()
            
            # Update revenue metrics
            amount_usd = invoice['amount_paid'] / 100.0
            revenue_metrics.labels(period='monthly', tier='unknown').inc(amount_usd)
            
            billing_events.labels(event_type='payment_succeeded', customer=invoice['customer']).inc()
            
        except Exception as e:
            logger.error(f"Error handling payment succeeded: {e}")
    
    async def generate_usage_analytics(self, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive usage analytics"""
        with self.Session() as session:
            # Base query
            query = session.query(UsageRecordDB)
            
            if customer_id:
                query = query.filter(UsageRecordDB.customer_id == customer_id)
            
            # Time-based analysis
            now = datetime.utcnow()
            last_30_days = now - timedelta(days=30)
            
            usage_data = query.filter(UsageRecordDB.timestamp >= last_30_days).all()
            
            # Aggregate by customer and usage type
            analytics = {
                'total_customers': len(set(record.customer_id for record in usage_data)),
                'total_requests': sum(record.quantity for record in usage_data if record.usage_type == UsageType.API_REQUEST.value),
                'usage_by_type': {},
                'usage_by_customer': {},
                'daily_usage': {},
                'top_customers': []
            }
            
            # Group by usage type
            for usage_type in UsageType:
                type_usage = [r for r in usage_data if r.usage_type == usage_type.value]
                analytics['usage_by_type'][usage_type.value] = {
                    'total': sum(r.quantity for r in type_usage),
                    'unique_customers': len(set(r.customer_id for r in type_usage))
                }
            
            # Group by customer
            customer_usage = {}
            for record in usage_data:
                if record.customer_id not in customer_usage:
                    customer_usage[record.customer_id] = 0
                customer_usage[record.customer_id] += record.quantity
            
            analytics['usage_by_customer'] = customer_usage
            analytics['top_customers'] = sorted(
                customer_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Daily usage trend
            daily_usage = {}
            for record in usage_data:
                day = record.timestamp.date()
                if day not in daily_usage:
                    daily_usage[day] = 0
                daily_usage[day] += record.quantity
            
            analytics['daily_usage'] = {
                str(day): usage for day, usage in sorted(daily_usage.items())
            }
            
            return analytics
    
    async def upgrade_subscription(self, customer_id: str, new_tier: SubscriptionTier) -> Dict[str, Any]:
        """Upgrade customer subscription"""
        try:
            customer_info = await self._get_customer_info(customer_id)
            
            if not customer_info:
                raise HTTPException(status_code=404, detail="Customer not found")
            
            current_tier = SubscriptionTier(customer_info['tier'])
            
            if new_tier == current_tier:
                raise HTTPException(status_code=400, detail="Already on requested tier")
            
            # Get new price
            new_tier_config = BillingSystemConfig.TIERS[new_tier]
            new_price_id = await self._get_or_create_price(
                new_tier,
                BillingPeriod(customer_info['billing_period']),
                new_tier_config.base_price
            )
            
            # Update Stripe subscription
            stripe_subscription = stripe.Subscription.retrieve(customer_info['stripe_subscription_id'])
            
            stripe.Subscription.modify(
                customer_info['stripe_subscription_id'],
                items=[{
                    'id': stripe_subscription['items']['data'][0]['id'],
                    'price': new_price_id
                }],
                proration_behavior='always_invoice'
            )
            
            # Update database
            with self.Session() as session:
                # Update customer
                customer = session.query(CustomerDB).filter_by(id=customer_id).first()
                customer.subscription_tier = new_tier.value
                
                # Update subscription
                subscription = session.query(SubscriptionDB).filter_by(customer_id=customer_id).first()
                subscription.tier = new_tier.value
                
                session.commit()
            
            # Update metrics
            subscription_count.labels(tier=current_tier.value).dec()
            subscription_count.labels(tier=new_tier.value).inc()
            billing_events.labels(event_type='subscription_upgraded', customer=customer_id).inc()
            
            return {
                'customer_id': customer_id,
                'old_tier': current_tier.value,
                'new_tier': new_tier.value,
                'status': 'upgraded'
            }
            
        except Exception as e:
            logger.error(f"Error upgrading subscription for {customer_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to upgrade subscription")
    
    async def _get_customer_info(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer information from database"""
        with self.Session() as session:
            customer = session.query(CustomerDB).filter_by(id=customer_id).first()
            subscription = session.query(SubscriptionDB).filter_by(customer_id=customer_id).first()
            
            if not customer or not subscription:
                return None
            
            return {
                'customer_id': customer.id,
                'email': customer.email,
                'tier': customer.subscription_tier,
                'billing_period': customer.billing_period,
                'stripe_customer_id': customer.stripe_customer_id,
                'stripe_subscription_id': subscription.stripe_subscription_id,
                'current_period_start': subscription.current_period_start,
                'current_period_end': subscription.current_period_end,
                'status': subscription.status
            }
    
    async def _get_period_usage(self, customer_id: str, usage_type: UsageType,
                              period_start: datetime, period_end: datetime) -> int:
        """Get usage for specific period"""
        with self.Session() as session:
            result = session.query(UsageRecordDB).filter(
                UsageRecordDB.customer_id == customer_id,
                UsageRecordDB.usage_type == usage_type.value,
                UsageRecordDB.timestamp >= period_start,
                UsageRecordDB.timestamp <= period_end
            ).all()
            
            return sum(record.quantity for record in result)
    
    async def _get_or_create_price(self, tier: SubscriptionTier, period: BillingPeriod, amount: Decimal) -> str:
        """Get or create Stripe price for tier and period"""
        price_lookup_key = f"{tier.value}_{period.value}"
        
        try:
            # Try to find existing price
            prices = stripe.Price.list(lookup_keys=[price_lookup_key])
            if prices.data:
                return prices.data[0].id
        except:
            pass
        
        # Create new price
        interval = 'month' if period == BillingPeriod.MONTHLY else 'year'
        
        price = stripe.Price.create(
            unit_amount=int(amount * 100),  # Convert to cents
            currency='usd',
            recurring={'interval': interval},
            product_data={
                'name': f"NBA Analytics {BillingSystemConfig.TIERS[tier].name} Plan"
            },
            lookup_key=price_lookup_key
        )
        
        return price.id
    
    async def _get_customer_id_from_stripe_customer(self, stripe_customer_id: str) -> str:
        """Get internal customer ID from Stripe customer ID"""
        with self.Session() as session:
            customer = session.query(CustomerDB).filter_by(
                stripe_customer_id=stripe_customer_id
            ).first()
            
            return customer.id if customer else ""


# FastAPI endpoints for billing system
app = FastAPI(title="NBA Analytics Billing System")

billing_system = None

@app.on_event("startup")
async def startup():
    global billing_system
    config = {
        'stripe_secret_key': 'sk_test_...',
        'stripe_webhook_secret': 'whsec_...',
        'postgres_url': 'postgresql://user:pass@localhost/billing',
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    billing_system = BillingSystem(config)

@app.post("/customers")
async def create_customer(customer_data: CustomerCreate):
    """Create new customer"""
    return await billing_system.create_customer(customer_data)

@app.post("/usage")
async def record_usage(usage: UsageRecord):
    """Record usage for billing"""
    await billing_system.record_usage(usage)
    return {"status": "recorded"}

@app.get("/customers/{customer_id}/usage")
async def get_usage_summary(customer_id: str):
    """Get customer usage summary"""
    return await billing_system.get_customer_usage_summary(customer_id)

@app.post("/customers/{customer_id}/upgrade")
async def upgrade_subscription(customer_id: str, new_tier: SubscriptionTier):
    """Upgrade customer subscription"""
    return await billing_system.upgrade_subscription(customer_id, new_tier)

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    return await billing_system.handle_stripe_webhook(request)

@app.get("/analytics/usage")
async def usage_analytics(customer_id: Optional[str] = None):
    """Get usage analytics"""
    return await billing_system.generate_usage_analytics(customer_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)