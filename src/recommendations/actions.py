"""
Actionable business recommendations and retention engine for SEGMENTIQ.

Maps customer lifecycle segments to concrete marketing, retention, and upsell
strategies tailored for an e-commerce context (specializing in anime merchandise,
collectibles, and pop-culture retail).
"""

from copy import deepcopy
from typing import Any, Sequence
import pandas as pd

# Priority hierarchy mapping for sorting
PRIORITY_ORDER = {"Urgent": 0, "High": 1, "Medium": 2, "Low": 3}

# Catalog of marketing and retention strategies per customer segment
SEGMENT_STRATEGIES: dict[str, dict[str, Any]] = {
    "Champions": {
        "action_title": "VIP Loyalty Rewards & Exclusive Drops",
        "priority": "High",
        "channel": ["Email", "In-App Notification", "Exclusive Discord / VIP Club"],
        "recommended_offers": [
            "Early access reservation window for limited-edition scale figures",
            "Double VIP loyalty points on all Japanese import pre-orders",
            "Exclusive invitations to brand ambassador and community referral programs",
            "Complimentary collector art print or acrylic stand on high-tier orders",
        ],
        "business_goal": "Reward brand evangelists, drive social proof/advocacy, and maximize customer lifetime value (LTV).",
    },
    "Loyal Customers": {
        "action_title": "Franchise Cross-Sell & Loyalty Retention",
        "priority": "High",
        "channel": ["Email", "Push Notification", "In-App"],
        "recommended_offers": [
            "Curated franchise bundle discounts (e.g., Manga + Nendoroid sets)",
            "Exclusive loyalty reward tiers with free domestic standard shipping",
            "Personalized review and unboxing photo reward incentives",
            "Personalized pre-order discounts for upcoming anime season adaptations",
        ],
        "business_goal": "Increase average order value (AOV) and maintain steady purchase cadence across product categories.",
    },
    "Potential Loyalists": {
        "action_title": "Membership Upsell & Category Expansion",
        "priority": "Medium",
        "channel": ["Email", "Push Notification"],
        "recommended_offers": [
            "10% off second order incentive valid for 14 days",
            "Explore trending apparel, plushies, and vinyl figures showcase",
            "Invite to join the SEGMENTIQ Collector Club with welcome bonus points",
            "Free mystery gachapon collectible on qualifying orders over £40",
        ],
        "business_goal": "Nurture occasional purchasers into recurring, multi-category brand loyalists.",
    },
    "New Customers": {
        "action_title": "Welcome Onboarding & Second-Order Activation",
        "priority": "Medium",
        "channel": ["Email", "In-App Welcome Banner"],
        "recommended_offers": [
            "15% welcome discount voucher on the next store purchase",
            "Collector starter guide: How to identify authentic official merchandise",
            "Curated top-10 bestsellers across popular anime series",
            "Clear explanation of the 30-day hassle-free return and damage warranty",
        ],
        "business_goal": "Build initial brand trust and secure the crucial second transaction.",
    },
    "Recent Customers": {
        "action_title": "Welcome Onboarding & Second-Order Activation",
        "priority": "Medium",
        "channel": ["Email", "In-App Welcome Banner"],
        "recommended_offers": [
            "15% welcome discount voucher on the next store purchase",
            "Collector starter guide: How to identify authentic official merchandise",
            "Curated top-10 bestsellers across popular anime series",
            "Clear explanation of the 30-day hassle-free return and damage warranty",
        ],
        "business_goal": "Build initial brand trust and secure the crucial second transaction.",
    },
    "Promising": {
        "action_title": "Franchise Discovery & Starter Incentives",
        "priority": "Medium",
        "channel": ["Email", "Push Notification"],
        "recommended_offers": [
            "Personalized recommendations based on recently viewed series",
            "Limited-time free shipping threshold lowered to £25",
            "Complimentary sticker pack and keychain with next checkout",
            "Flash deals on light novels and budget prize figures",
        ],
        "business_goal": "Deepen product discovery and boost brand affinity with low-friction offers.",
    },
    "Customers Needing Attention": {
        "action_title": "Time-Sensitive Re-Engagement",
        "priority": "High",
        "channel": ["Email", "SMS", "Push Notification"],
        "recommended_offers": [
            "Limited-time £10 voucher code expiring in 5 days",
            "New arrival alerts for favorite merchandise categories",
            "Priority restock reservation for sold-out favorites",
            "Free premium collectible display case on orders over £75",
        ],
        "business_goal": "Re-activate slowing buyers before they slip into the At-Risk cohort.",
    },
    "About to Sleep": {
        "action_title": "Reactivation Drip & 'We Miss You' Deals",
        "priority": "Medium",
        "channel": ["Email", "SMS"],
        "recommended_offers": [
            "'We miss you!' 15% discount code tailored to past franchises",
            "Seasonal anime merchandise digest featuring newly announced releases",
            "Quick 1-question feedback survey with an instant gift card reward",
            "Curated clearance section access before public release",
        ],
        "business_goal": "Re-engage slipping customers before complete disengagement and churn.",
    },
    "At Risk": {
        "action_title": "Aggressive Win-Back Revival Campaign",
        "priority": "Urgent",
        "channel": ["Email", "SMS", "Targeted Push Notification"],
        "recommended_offers": [
            "High-incentive 25% win-back revival coupon code",
            "Highlight major store expansions and newly secured official licensing partners",
            "Free express shipping voucher on any order",
            "Guaranteed allocation for upcoming major anime convention exclusives",
        ],
        "business_goal": "Recover high-value historical spenders on the immediate verge of permanent churn.",
    },
    "Can't Lose Them": {
        "action_title": "Executive VIP Win-Back & Direct Outreach",
        "priority": "Urgent",
        "channel": ["Direct Contact", "Email", "Personalized SMS"],
        "recommended_offers": [
            "Personal concierge email or customer-care direct outreach",
            "Complimentary high-value collector gift box with any renewal order",
            "Lifetime VIP membership tier status upgrade",
            "Exclusive invitation to pre-screen rare vaulted imports",
        ],
        "business_goal": "Immediately halt churn among our historical top-tier whales and highest-spending collectors.",
    },
    "Hibernating": {
        "action_title": "Low-Cost Clearance Outlet Reactivation",
        "priority": "Low",
        "channel": ["Email"],
        "recommended_offers": [
            "Deep-discount outlet and warehouse clearance sale notifications",
            "Up to 50% off discontinued apparel and older anime series stock",
            "Bi-annual mega-clearance digest",
        ],
        "business_goal": "Drive sporadic revival purchases with zero marginal ad spend.",
    },
    "Lost": {
        "action_title": "Sunset Protocol & Exit Survey",
        "priority": "Low",
        "channel": ["Email"],
        "recommended_offers": [
            "Short 1-question exit survey to identify root churn causes (price, shipping, catalog)",
            "Annual Black Friday / Anniversary blowout promotion notice",
            "Option to pause emails or select specific anime franchises only",
        ],
        "business_goal": "Gather valuable churn intelligence while protecting domain email deliverability and sender reputation.",
    },
    "Others": {
        "action_title": "General Brand Engagement & Seasonal Catalog",
        "priority": "Low",
        "channel": ["Email"],
        "recommended_offers": [
            "General bi-weekly anime culture newsletter and industry news",
            "Standard seasonal catalog releases (Spring/Summer/Fall/Winter seasons)",
            "Community fan-art contests and social media giveaways",
        ],
        "business_goal": "Maintain broad brand awareness and provide low-frequency touchpoints for unclassified users.",
    },
}

# Lookup aliases for case/naming variations
_SEGMENT_ALIASES: dict[str, str] = {
    "champions": "Champions",
    "loyal customers": "Loyal Customers",
    "loyal": "Loyal Customers",
    "potential loyalists": "Potential Loyalists",
    "potential loyalist": "Potential Loyalists",
    "new customers": "New Customers",
    "new customer": "New Customers",
    "recent customers": "Recent Customers",
    "recent customer": "Recent Customers",
    "recent customers / new customers": "New Customers",
    "promising": "Promising",
    "customers needing attention": "Customers Needing Attention",
    "needing attention": "Customers Needing Attention",
    "about to sleep": "About to Sleep",
    "at risk": "At Risk",
    "can't lose them": "Can't Lose Them",
    "cant lose them": "Can't Lose Them",
    "hibernating": "Hibernating",
    "lost": "Lost",
    "others": "Others",
    "other": "Others",
}


def recommend_actions_for_customer(segment: str) -> dict[str, Any]:
    """
    Retrieve marketing and retention recommendations for a single customer segment.

    Performs case-insensitive, whitespace-stripped lookup with graceful fallback
    to the 'Others' strategy if the segment is unrecognized or missing.

    Parameters:
        segment (str): Name of customer segment (e.g. 'Champions', 'At Risk').

    Returns:
        dict[str, Any]: Strategy dictionary containing:
            - 'action_title': str
            - 'priority': str ('Urgent', 'High', 'Medium', 'Low')
            - 'channel': list[str]
            - 'recommended_offers': list[str]
            - 'business_goal': str
    """
    if segment is None or not isinstance(segment, str):
        return deepcopy(SEGMENT_STRATEGIES["Others"])

    cleaned = segment.strip().lower()
    canonical_name = _SEGMENT_ALIASES.get(cleaned, "Others")

    strategy = SEGMENT_STRATEGIES.get(canonical_name, SEGMENT_STRATEGIES["Others"])
    return deepcopy(strategy)


def assign_recommendations(
    df: pd.DataFrame,
    segment_col: str = "Segment",
) -> pd.DataFrame:
    """
    Assign concrete marketing actions, priority levels, and channels to each customer.

    Non-destructive: operates on a copy of `df` and appends recommendation columns.

    Parameters:
        df (pd.DataFrame): Customer DataFrame containing `segment_col`.
        segment_col (str): Column name containing segment classifications. Defaults to 'Segment'.

    Returns:
        pd.DataFrame: Enriched DataFrame copy with additional columns:
            - 'Action_Title': str
            - 'Action_Priority': str
            - 'Action_Goal': str
            - 'Recommended_Channels': str (comma-separated channels)

    Raises:
        ValueError: If df is None or segment_col is missing from df.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if segment_col not in df.columns:
        raise ValueError(
            f"Required segment column '{segment_col}' not found in DataFrame. "
            f"Found columns: {list(df.columns)}"
        )

    result_df = df.copy()

    if result_df.empty:
        result_df["Action_Title"] = pd.Series(dtype="str")
        result_df["Action_Priority"] = pd.Series(dtype="str")
        result_df["Action_Goal"] = pd.Series(dtype="str")
        result_df["Recommended_Channels"] = pd.Series(dtype="str")
        return result_df

    action_titles: list[str] = []
    priorities: list[str] = []
    goals: list[str] = []
    channels: list[str] = []

    for seg in result_df[segment_col]:
        strategy = recommend_actions_for_customer(str(seg) if pd.notna(seg) else "")
        action_titles.append(strategy["action_title"])
        priorities.append(strategy["priority"])
        goals.append(strategy["business_goal"])
        channels.append(", ".join(strategy["channel"]))

    result_df["Action_Title"] = action_titles
    result_df["Action_Priority"] = priorities
    result_df["Action_Goal"] = goals
    result_df["Recommended_Channels"] = channels

    return result_df


def get_action_plan_summary(
    df: pd.DataFrame,
    segment_col: str = "Segment",
) -> pd.DataFrame:
    """
    Generate an executive action plan summary aggregated at the segment level.

    Computes customer counts, percentage distribution, mapped action priority,
    strategy title, and primary business goal, sorted logically by priority
    ('Urgent' -> 'High' -> 'Medium' -> 'Low') and customer count descending.

    Parameters:
        df (pd.DataFrame): Clustered or segmented DataFrame containing `segment_col`.
        segment_col (str): Column name containing customer segments. Defaults to 'Segment'.

    Returns:
        pd.DataFrame: Action plan summary table with columns:
            ['Segment', 'Count', 'Percentage', 'Action_Priority', 'Action_Title', 'Action_Goal'].

    Raises:
        ValueError: If df is None or segment_col is not present.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if segment_col not in df.columns:
        raise ValueError(
            f"Segment column '{segment_col}' not found in DataFrame. "
            f"Found columns: {list(df.columns)}"
        )

    summary_columns = [
        "Segment",
        "Count",
        "Percentage",
        "Action_Priority",
        "Action_Title",
        "Action_Goal",
    ]

    if df.empty:
        return pd.DataFrame(columns=summary_columns)

    total_customers = len(df)
    grouped = df.groupby(segment_col, observed=False).size()

    summary_rows = []
    for segment, count in grouped.items():
        strategy = recommend_actions_for_customer(str(segment))
        pct = round((count / total_customers) * 100, 2)
        priority = strategy["priority"]
        summary_rows.append(
            {
                "Segment": segment,
                "Count": count,
                "Percentage": pct,
                "Action_Priority": priority,
                "Action_Title": strategy["action_title"],
                "Action_Goal": strategy["business_goal"],
                "_priority_rank": PRIORITY_ORDER.get(priority, 99),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Sort primarily by Priority ('Urgent' -> 'High' -> 'Medium' -> 'Low')
    # and secondarily by Count descending
    summary_df = summary_df.sort_values(
        by=["_priority_rank", "Count"], ascending=[True, False]
    ).drop(columns=["_priority_rank"]).reset_index(drop=True)

    return summary_df


__all__ = [
    "recommend_actions_for_customer",
    "assign_recommendations",
    "get_action_plan_summary",
    "SEGMENT_STRATEGIES",
    "PRIORITY_ORDER",
]
