from .schemas import DBTFramework, DBTModule, DBTSkill, SubSkill


def create_dbt_framework() -> DBTFramework:
    """Create complete DBT framework with all modules and their skills"""

    # Mindfulness Module
    mindfulness_module = DBTModule(
        name="Mindfulness",
        description="Foundational skills for present-moment awareness and non-judgmental observation",
        skills=[
            # Core mindfulness skills
            DBTSkill(
                name="Observe",
                category="Foundational",
                description="Notice and pay attention to present moment experiences without trying to change them",
            ),
            DBTSkill(
                name="Describe",
                category="Foundational",
                description="Put words to experiences without judgment, interpretation, or opinion",
            ),
            DBTSkill(
                name="Participate",
                category="Foundational",
                description="Throw yourself completely into present moment activities",
            ),
            # How to practice mindfulness skills
            DBTSkill(
                name="Non-Judgmentally",
                category="Applied",
                description="Practice mindfulness without evaluating experiences as good or bad",
            ),
            DBTSkill(
                name="One-Mind",
                category="Applied",
                description="Focus completely on one thing at a time, in the moment",
            ),
            DBTSkill(
                name="Effectively",
                category="Applied",
                description="Focus on what works in the present situation to achieve your goals",
            ),
        ],
    )

    # Emotion Regulation Module
    emotion_regulation_module = DBTModule(
        name="Emotion Regulation",
        description="Skills for understanding, managing, and changing emotions effectively",
        skills=[
            DBTSkill(
                name="Understand Emotions",
                category="Foundational",
                description="Learn what emotions are, how they function, and how to identify them",
            ),
            DBTSkill(
                name="Check the Facts",
                category="Applied",
                description="Verify if your emotional response fits the actual facts of the situation",
            ),
            DBTSkill(
                name="Opposite Action",
                category="Applied",
                description="Act opposite to destructive emotional urges when emotions don't fit the facts",
            ),
            DBTSkill(
                name="Problem Solving",
                category="Applied",
                description="Take action to change situations that are causing unwanted emotions",
            ),
            DBTSkill(
                name="Accumulating Positives",
                category="Applied",
                description="Build positive experiences to increase emotional resilience",
                sub_skills=[
                    SubSkill(
                        name="Pleasant Events",
                        description="Schedule enjoyable short-term activities",
                    ),
                    SubSkill(
                        name="Building Mastery",
                        description="Engage in activities that create accomplishment",
                    ),
                    SubSkill(
                        name="Values-Based Living",
                        description="Live according to your personal values",
                    ),
                ],
            ),
            DBTSkill(
                name="Cope Ahead",
                category="Applied",
                description="Prepare and rehearse coping strategies for anticipated difficult situations",
            ),
        ],
    )

    # Distress Tolerance Module
    distress_tolerance_module = DBTModule(
        name="Distress Tolerance",
        description="Skills for surviving crisis situations and accepting reality when you cannot change it",
        skills=[
            DBTSkill(
                name="TIP",
                category="Crisis Management",
                description="Change body chemistry quickly to reduce extreme emotions",
                sub_skills=[
                    SubSkill(
                        name="Temperature",
                        description="Change body temperature with cold water on face/hands",
                    ),
                    SubSkill(
                        name="Intense Exercise",
                        description="Use vigorous physical activity for 10+ minutes",
                    ),
                    SubSkill(
                        name="Paced Breathing",
                        description="Slow exhale longer than inhale to activate calm",
                    ),
                ],
            ),
            DBTSkill(
                name="Distract",
                category="Crisis Management",
                description="Redirect attention away from distressing situations temporarily",
                sub_skills=[
                    SubSkill(
                        name="Activities",
                        description="Engage in pleasant or necessary activities",
                    ),
                    SubSkill(
                        name="Contributing", description="Help others or volunteer"
                    ),
                    SubSkill(
                        name="Comparisons",
                        description="Compare to less fortunate or past coping",
                    ),
                    SubSkill(
                        name="Emotions",
                        description="Create different emotions with music, comedy, etc.",
                    ),
                    SubSkill(
                        name="Push Away",
                        description="Mentally push the situation away temporarily",
                    ),
                    SubSkill(
                        name="Thoughts",
                        description="Fill your mind with other thoughts",
                    ),
                    SubSkill(
                        name="Sensations",
                        description="Use intense sensations like ice cubes or hot shower",
                    ),
                ],
            ),
            DBTSkill(
                name="Self-Soothe",
                category="Crisis Management",
                description="Comfort yourself through the five senses during distress",
                sub_skills=[
                    SubSkill(
                        name="Vision", description="Look at beautiful or calming things"
                    ),
                    SubSkill(
                        name="Hearing", description="Listen to soothing sounds or music"
                    ),
                    SubSkill(name="Smell", description="Use pleasant scents or aromas"),
                    SubSkill(
                        name="Taste", description="Eat or drink something comforting"
                    ),
                    SubSkill(
                        name="Touch", description="Use soothing physical sensations"
                    ),
                ],
            ),
            DBTSkill(
                name="Pros and Cons",
                category="Applied",
                description="Weigh advantages and disadvantages of acting on crisis urges vs. tolerating distress",
            ),
            DBTSkill(
                name="Radical Acceptance",
                category="Applied",
                description="Completely accept reality as it is, without approval or fighting against it",
            ),
        ],
    )

    # Interpersonal Effectiveness Module
    interpersonal_effectiveness_module = DBTModule(
        name="Interpersonal Effectiveness",
        description="Skills for asking for what you need, saying no, and maintaining relationships",
        skills=[
            DBTSkill(
                name="DEAR MAN",
                category="Applied",
                description="Technique for making requests and getting what you want",
                sub_skills=[
                    SubSkill(
                        name="Describe", description="Describe the situation with facts"
                    ),
                    SubSkill(
                        name="Express", description="Express your feelings and opinions"
                    ),
                    SubSkill(name="Assert", description="Assert your request clearly"),
                    SubSkill(
                        name="Reinforce",
                        description="Reinforce the benefits of getting what you want",
                    ),
                    SubSkill(name="Mindful", description="Stay focused on your goal"),
                    SubSkill(
                        name="Appear Confident",
                        description="Use confident body language and tone",
                    ),
                    SubSkill(name="Negotiate", description="Be willing to compromise"),
                ],
            ),
            DBTSkill(
                name="GIVE",
                category="Relationship Building",
                description="Maintain relationships while getting what you want",
                sub_skills=[
                    SubSkill(
                        name="Gentle", description="Be kind and respectful, no attacks"
                    ),
                    SubSkill(
                        name="Interested",
                        description="Listen and show genuine interest",
                    ),
                    SubSkill(
                        name="Validate",
                        description="Acknowledge the other person's perspective",
                    ),
                    SubSkill(
                        name="Easy Manner",
                        description="Use humor and be lighthearted when appropriate",
                    ),
                ],
            ),
            DBTSkill(
                name="FAST",
                category="Relationship Building",
                description="Maintain self-respect in interpersonal situations",
                sub_skills=[
                    SubSkill(name="Fair", description="Be fair to yourself and others"),
                    SubSkill(
                        name="Apologies",
                        description="Don't over-apologize or apologize for having opinions",
                    ),
                    SubSkill(
                        name="Stick to Values",
                        description="Don't compromise your values for acceptance",
                    ),
                    SubSkill(
                        name="Truthful",
                        description="Be honest and don't lie or act helpless",
                    ),
                ],
            ),
        ],
    )

    return DBTFramework(
        modules=[
            mindfulness_module,
            emotion_regulation_module,
            distress_tolerance_module,
            interpersonal_effectiveness_module,
        ]
    )


# Example usage
if __name__ == "__main__":
    framework = create_dbt_framework()

    # Example: Get all skills in Mindfulness module
    mindfulness = framework.get_module_by_name("Mindfulness")
    if mindfulness:
        print(f"{mindfulness.name} Module:")
        print(f"  Description: {mindfulness.description}")
        print("  Skills:")
        for skill in mindfulness.skills:
            print(f"    - {skill.name} ({skill.category}): {skill.description}")

    # Example: Get all Crisis Management skills across all modules
    crisis_skills = framework.get_skills_by_category("Crisis Management")
    print("\nAll Crisis Management Skills:")
    for skill in crisis_skills:
        print(f"  - {skill.name}: {skill.description}")
        if skill.sub_skills:
            for sub_skill in skill.sub_skills:
                print(f"    • {sub_skill.name}: {sub_skill.description}")

    # Example: Find a specific skill
    tip_skill = None
    for module in framework.modules:
        tip_skill = module.get_skill_by_name("TIP")
        if tip_skill:
            break

    if tip_skill:
        print("\nTIP Skill Details:")
        print(f"  Description: {tip_skill.description}")
        print("  Sub-skills:")
        for sub_skill in tip_skill.sub_skills:
            print(f"    • {sub_skill.name}: {sub_skill.description}")
