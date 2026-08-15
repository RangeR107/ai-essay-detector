"""
Curated mapping from the Gemini AI-essay batch's 50 real generation
categories (500_admissions_prompts.csv) to the plan's 7 personal-statement
themes. Built to fix a specific, measured problem: the old approach ran
the same keyword heuristic used for human essays over the AI essays too,
even though the AI essays already carry a KNOWN true category — that
heuristic didn't map cleanly onto Gemini's writing register and produced
a lopsided result (~89% of AI essays landing in just 2 of 7 themes,
data/README.md). Since ground truth exists here, use it instead of
guessing from text.

Every one of the 50 categories is assigned to exactly one theme (a
judgment call where a category could plausibly fit more than one — kept
to a single best fit rather than splitting, for simplicity). See
docs/IMPLEMENTATION.md for the before/after theme distribution this
produced.
"""
from __future__ import annotations

CATEGORY_TO_THEME: dict[str, str] = {
    # background_identity
    "Identity & self-understanding": "background_identity",
    "Family & relationships": "background_identity",
    "Community & belonging": "background_identity",
    "Culture & heritage": "background_identity",
    "Language & communication": "background_identity",
    "Place & home": "background_identity",
    "Belonging & inclusion": "background_identity",
    # obstacle_setback
    "Challenges & resilience": "obstacle_setback",
    "Failure & mistakes": "obstacle_setback",
    "Health & wellbeing": "obstacle_setback",
    "Adaptability": "obstacle_setback",
    # challenging_belief
    "Values & ethics": "challenging_belief",
    "Perspective & disagreement": "challenging_belief",
    "Perspective & self-reflection": "challenging_belief",
    "Questions & uncertainty": "challenging_belief",
    "Personal philosophy": "challenging_belief",
    # gratitude
    "Mentors & role models": "gratitude",
    "Service & helping others": "gratitude",
    # growth_accomplishment
    "Growth & change": "growth_accomplishment",
    "Leadership": "growth_accomplishment",
    "Work & responsibility": "growth_accomplishment",
    "Decision-making": "growth_accomplishment",
    "Independence": "growth_accomplishment",
    "Patience & persistence": "growth_accomplishment",
    "Future & aspirations": "growth_accomplishment",
    "College readiness & transition": "growth_accomplishment",
    "Personal impact": "growth_accomplishment",
    "Courage & risk": "growth_accomplishment",
    # captivating_topic
    "Curiosity & learning": "captivating_topic",
    "Academic life": "captivating_topic",
    "Creativity": "captivating_topic",
    "Technology & digital life": "captivating_topic",
    "Science & discovery": "captivating_topic",
    "Arts & expression": "captivating_topic",
    "Hobbies & everyday passions": "captivating_topic",
    "Imagination & hypothetical thinking": "captivating_topic",
    "Books & reading": "captivating_topic",
    "Music": "captivating_topic",
    # open_topic (fallback bucket, but these genuinely don't fit the other 6 well)
    "Time & memory": "open_topic",
    "Sports & competition": "open_topic",
    "Observation & small details": "open_topic",
    "Social impact": "open_topic",
    "Environment & sustainability": "open_topic",
    "Travel & unfamiliar experiences": "open_topic",
    "Humor & personality": "open_topic",
    "Meaningful objects": "open_topic",
    "Food & memory": "open_topic",
    "Collaboration": "open_topic",
    "Rules & conventions": "open_topic",
    "Future society": "open_topic",
}


def theme_for_category(category: str) -> str | None:
    return CATEGORY_TO_THEME.get(category.strip())
