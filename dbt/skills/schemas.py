from typing import Literal, Optional

from pydantic import BaseModel

# Core module types
Module = Literal[
    "Mindfulness",
    "Emotion Regulation",
    "Distress Tolerance",
    "Interpersonal Effectiveness",
]

# Skill categories for better organization
SkillCategory = Literal[
    "Foundational", "Applied", "Crisis Management", "Relationship Building"
]


class SubSkill(BaseModel):
    """Individual components within a skill"""

    name: str
    description: Optional[str] = None


class DBTSkill(BaseModel):
    """A specific skill with potential sub-components"""

    name: str
    description: Optional[str] = None
    sub_skills: list[SubSkill] = []
    category: Optional[SkillCategory] = None


class DBTModule(BaseModel):
    """Complete module with its associated skills"""

    name: Module
    description: str
    skills: list[DBTSkill]

    def get_skills_by_category(self, category: SkillCategory) -> list[DBTSkill]:
        """Get all skills in this module filtered by category"""
        return [skill for skill in self.skills if skill.category == category]

    def get_skill_by_name(self, skill_name: str) -> Optional[DBTSkill]:
        """Get a specific skill by name"""
        for skill in self.skills:
            if skill.name.lower() == skill_name.lower():
                return skill
        return None


class DBTFramework(BaseModel):
    """Complete DBT framework with all modules"""

    modules: list[DBTModule]

    def get_module_by_name(self, module_name: Module) -> Optional[DBTModule]:
        """Get a specific module by name"""
        for module in self.modules:
            if module.name == module_name:
                return module
        return None

    def get_all_skills(self) -> list[DBTSkill]:
        """Get all skills across all modules"""
        all_skills = []
        for module in self.modules:
            all_skills.extend(module.skills)
        return all_skills

    def get_skills_by_category(self, category: SkillCategory) -> list[DBTSkill]:
        """Get all skills across all modules filtered by category"""
        return [skill for skill in self.get_all_skills() if skill.category == category]
