from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# Import observability tools from main.py
try:
    from openinference.instrumentation import using_prompt_template
except Exception:
    def using_prompt_template(**kwargs):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()


class MealPlanRequest(BaseModel):
    dietary_preference: str  # vegetarian, vegan, gluten-free, etc.
    duration: str  # "1 week", "3 days", etc.
    budget: Optional[str] = "moderate"
    allergies: Optional[str] = None
    cuisine_preferences: Optional[str] = None
    cooking_time: Optional[str] = "30-60 minutes"  # preferred cooking time per meal
    family_size: Optional[int] = 2


class MealPlanResponse(BaseModel):
    result: str
    grocery_list: Optional[str] = None
    nutritional_info: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []


class MealPlanState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    meal_request: Dict[str, Any]
    research: Optional[str]  # Recipe research
    nutrition: Optional[str]  # Nutritional analysis
    planning: Optional[str]  # Meal scheduling
    shopping: Optional[str]  # Grocery list
    final: Optional[str]  # Final meal plan
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


# Meal planning tools
@tool
def recipe_search(dietary_preference: str, cuisine: Optional[str] = None) -> str:
    """Search for recipes based on dietary preferences and cuisine type."""
    cuisine_str = f" focusing on {cuisine} cuisine" if cuisine else ""
    return f"""Recipe suggestions for {dietary_preference} diet{cuisine_str}:
    
    Breakfast Options:
    - Overnight oats with fruits and nuts (prep time: 5 min, cook time: 0 min)
    - Avocado toast with seeds (prep time: 10 min, cook time: 5 min)
    - Smoothie bowl with plant protein (prep time: 15 min, cook time: 0 min)
    - Chia pudding with berries (prep time: 10 min, cook time: 0 min)
    
    Lunch Options:
    - Buddha bowl with quinoa and roasted vegetables (prep time: 20 min, cook time: 30 min)
    - Lentil soup with crusty bread (prep time: 15 min, cook time: 45 min)
    - Chickpea salad wrap (prep time: 20 min, cook time: 0 min)
    - Veggie stir-fry with tofu (prep time: 15 min, cook time: 20 min)
    
    Dinner Options:
    - Black bean tacos with avocado (prep time: 25 min, cook time: 15 min)
    - Pasta primavera with seasonal vegetables (prep time: 20 min, cook time: 25 min)
    - Stuffed bell peppers with rice (prep time: 30 min, cook time: 45 min)
    - Thai curry with coconut milk and vegetables (prep time: 25 min, cook time: 30 min)"""


@tool
def nutritional_analysis(dietary_preference: str, duration: str) -> str:
    """Analyze nutritional requirements for the given dietary preference and duration."""
    return f"""Nutritional analysis for {dietary_preference} diet over {duration}:
    
    Daily Macronutrient Targets:
    - Protein: 50-70g (legumes, nuts, seeds, plant proteins)
    - Carbohydrates: 225-325g (whole grains, fruits, vegetables)
    - Healthy Fats: 44-78g (avocados, nuts, olive oil)
    - Fiber: 25-35g (vegetables, fruits, whole grains)
    
    Key Nutrients to Monitor:
    - Vitamin B12: Consider fortified foods or supplements
    - Iron: Combine with vitamin C for better absorption
    - Calcium: Dark leafy greens, tahini, fortified plant milks
    - Omega-3s: Flaxseeds, chia seeds, walnuts
    - Zinc: Pumpkin seeds, cashews, chickpeas
    
    Weekly Meal Balance:
    - 21 meals total (3 meals × 7 days)
    - Include variety of protein sources across meals
    - Ensure 5-7 servings of fruits and vegetables daily
    - Incorporate whole grains in 2-3 meals per day
    - Balance cooking methods: raw, steamed, roasted, sautéed"""


@tool
def meal_scheduling(duration: str, cooking_time: str, family_size: int = 2) -> str:
    """Create a meal schedule based on duration, cooking time preferences, and family size."""
    return f"""Meal schedule for {duration} (family size: {family_size}, preferred cooking time: {cooking_time}):
    
    Weekly Structure:
    Day 1 (Monday):
    - Breakfast: Quick overnight oats (5 min prep)
    - Lunch: Buddha bowl with quinoa (50 min total)
    - Dinner: Black bean tacos (40 min total)
    
    Day 2 (Tuesday):
    - Breakfast: Smoothie bowl (15 min prep)
    - Lunch: Chickpea salad wrap (20 min prep)
    - Dinner: Pasta primavera (45 min total)
    
    Day 3 (Wednesday):
    - Breakfast: Avocado toast (15 min total)
    - Lunch: Lentil soup (60 min total - make large batch)
    - Dinner: Veggie stir-fry (35 min total)
    
    Day 4 (Thursday):
    - Breakfast: Chia pudding (10 min prep)
    - Lunch: Leftover lentil soup (5 min reheat)
    - Dinner: Stuffed bell peppers (75 min total)
    
    Day 5 (Friday):
    - Breakfast: Overnight oats (5 min prep)
    - Lunch: Buddha bowl variation (50 min total)
    - Dinner: Thai curry (55 min total)
    
    Weekend Meal Prep Tips:
    - Batch cook grains and legumes
    - Pre-chop vegetables for the week
    - Prepare overnight oats for 3 days
    - Make large batch of soup or curry"""


@tool
def grocery_list_generator(meals: str, family_size: int = 2, duration: str = "1 week") -> str:
    """Generate a categorized grocery list based on planned meals."""
    return f"""Grocery list for {duration} ({family_size} people):
    
    PRODUCE SECTION:
    - Avocados (4-6 pieces)
    - Bananas (6-8 pieces)
    - Berries (2 containers)
    - Bell peppers (6-8 pieces)
    - Leafy greens (2 bags spinach/kale)
    - Tomatoes (4-6 pieces)
    - Onions (2-3 pieces)
    - Garlic (1 bulb)
    - Carrots (1 bag)
    - Broccoli (2 heads)
    
    PANTRY ITEMS:
    - Quinoa (1 bag, 2 lbs)
    - Brown rice (1 bag, 2 lbs)
    - Rolled oats (1 container)
    - Black beans (3 cans)
    - Chickpeas (3 cans)
    - Lentils (2 bags, dry)
    - Coconut milk (2 cans)
    - Olive oil (1 bottle)
    - Tahini (1 jar)
    - Chia seeds (1 bag)
    
    REFRIGERATED:
    - Plant-based milk (2 cartons)
    - Tofu (2 blocks)
    - Nutritional yeast (1 container)
    - Whole grain bread (2 loaves)
    - Tortillas (1 package)
    
    FROZEN:
    - Mixed vegetables (2 bags)
    - Berries (1 bag)
    
    SPICES & CONDIMENTS:
    - Curry powder
    - Cumin
    - Turmeric
    - Sea salt
    - Black pepper
    - Lemon juice
    - Balsamic vinegar"""


@tool
def ingredient_substitutions(dietary_preference: str, allergies: Optional[str] = None) -> str:
    """Provide ingredient substitution suggestions for dietary restrictions and allergies."""
    allergy_note = f" (avoiding {allergies})" if allergies else ""
    return f"""Ingredient substitutions for {dietary_preference} diet{allergy_note}:
    
    Protein Substitutions:
    - Instead of meat: lentils, chickpeas, black beans, tofu, tempeh
    - Instead of eggs: flax eggs (1 tbsp ground flax + 3 tbsp water)
    - Instead of dairy protein: plant-based protein powder, hemp seeds
    
    Dairy Substitutions:
    - Instead of milk: oat milk, almond milk, soy milk, coconut milk
    - Instead of cheese: nutritional yeast, cashew cream, vegan cheese
    - Instead of butter: coconut oil, avocado, olive oil
    
    Common Allergy Substitutions:
    - Gluten-free: quinoa instead of wheat, rice noodles instead of pasta
    - Nut-free: sunflower seeds instead of almonds, tahini instead of peanut butter
    - Soy-free: coconut aminos instead of soy sauce, hemp protein instead of soy protein
    
    Flavor Enhancers:
    - Umami: mushrooms, miso paste, nutritional yeast, tamari
    - Sweetness: dates, maple syrup, applesauce
    - Creaminess: cashew cream, avocado, coconut cream"""


@tool
def cooking_tips(cooking_time: str, family_size: int = 2) -> str:
    """Provide cooking tips and time-saving strategies."""
    return f"""Cooking tips for {cooking_time} meal prep (family size: {family_size}):
    
    Time-Saving Strategies:
    - Batch cook grains and legumes on weekends
    - Use one-pot meals to minimize cleanup
    - Pre-chop vegetables when you get home from shopping
    - Keep a well-stocked pantry of staples
    
    Quick Cooking Methods:
    - Pressure cooker: reduces legume cooking time by 70%
    - Sheet pan meals: cook protein and vegetables together
    - Stir-frying: high heat, quick cooking for vegetables
    - Raw preparations: salads, overnight oats, smoothies
    
    Make-Ahead Options:
    - Overnight oats: prepare 3 days worth
    - Soup and curry: freeze portions for busy days
    - Grain bowls: prep components separately, assemble quickly
    - Energy balls: no-bake snacks ready in 20 minutes
    
    Kitchen Tools That Save Time:
    - Food processor for quick chopping
    - High-speed blender for smoothies
    - Rice cooker for perfect grains
    - Sharp knives for efficient prep"""


def _init_meal_llm():
    """Initialize LLM for meal planning (reuse logic from main.py)"""
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test meal plan"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


meal_llm = _init_meal_llm()


def research_agent(state: MealPlanState) -> MealPlanState:
    """Research agent for recipe discovery and ingredient information."""
    req = state["meal_request"]
    dietary_preference = req["dietary_preference"]
    cuisine = req.get("cuisine_preferences")
    
    prompt_t = (
        "You are a recipe research specialist.\n"
        "Find diverse, healthy recipes for {dietary_preference} diet.\n"
        "Cuisine preference: {cuisine}.\n"
        "Use tools to gather recipe information and ingredient details."
    )
    vars_ = {"dietary_preference": dietary_preference, "cuisine": cuisine or "varied"}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [recipe_search, ingredient_substitutions]
    agent = meal_llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    tool_results = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        tool_results = tr["messages"]
        
        messages.append(res)
        messages.extend(tool_results)
        messages.append(SystemMessage(content="Based on the recipe research, provide a comprehensive summary of meal options and ingredient information."))
        
        final_res = meal_llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def nutrition_agent(state: MealPlanState) -> MealPlanState:
    """Nutrition agent for dietary analysis and nutritional balance."""
    req = state["meal_request"]
    dietary_preference = req["dietary_preference"]
    duration = req["duration"]
    allergies = req.get("allergies")
    
    prompt_t = (
        "You are a plant-based nutrition expert.\n"
        "Analyze nutritional requirements for {dietary_preference} diet over {duration}.\n"
        "Consider allergies: {allergies}.\n"
        "Use tools to ensure nutritional adequacy and provide guidance."
    )
    vars_ = {"dietary_preference": dietary_preference, "duration": duration, "allergies": allergies or "none"}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [nutritional_analysis, ingredient_substitutions]
    agent = meal_llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "nutrition", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        messages.append(res)
        messages.extend(tr["messages"])
        messages.append(SystemMessage(content=f"Create a detailed nutritional analysis for {duration} of {dietary_preference} meals."))
        
        final_res = meal_llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "nutrition": out, "tool_calls": calls}


def planning_agent(state: MealPlanState) -> MealPlanState:
    """Planning agent for meal scheduling and time management."""
    req = state["meal_request"]
    duration = req["duration"]
    cooking_time = req.get("cooking_time", "30-60 minutes")
    family_size = req.get("family_size", 2)
    
    prompt_t = (
        "You are a meal planning coordinator.\n"
        "Create a practical meal schedule for {duration}.\n"
        "Cooking time preference: {cooking_time}.\n"
        "Family size: {family_size} people.\n"
        "Use tools to create an organized meal plan with timing."
    )
    vars_ = {"duration": duration, "cooking_time": cooking_time, "family_size": family_size}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [meal_scheduling, cooking_tips]
    agent = meal_llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "planning", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        messages.append(res)
        messages.extend(tr["messages"])
        messages.append(SystemMessage(content=f"Create a detailed {duration} meal plan with specific timing and preparation tips."))
        
        final_res = meal_llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "planning": out, "tool_calls": calls}


def shopping_agent(state: MealPlanState) -> MealPlanState:
    """Shopping agent for grocery list generation and organization."""
    req = state["meal_request"]
    duration = req["duration"]
    family_size = req.get("family_size", 2)
    budget = req.get("budget", "moderate")
    
    prompt_t = (
        "You are a grocery shopping specialist.\n"
        "Create an organized shopping list for {duration} of meals.\n"
        "Family size: {family_size}, Budget: {budget}.\n"
        "Use tools to generate a comprehensive, categorized grocery list."
    )
    vars_ = {"duration": duration, "family_size": family_size, "budget": budget}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [grocery_list_generator, cooking_tips]
    agent = meal_llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "shopping", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        messages.append(res)
        messages.extend(tr["messages"])
        messages.append(SystemMessage(content=f"Create a detailed, organized grocery list for {duration} of {family_size}-person meals within a {budget} budget."))
        
        final_res = meal_llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "shopping": out, "tool_calls": calls}


def meal_synthesis_agent(state: MealPlanState) -> MealPlanState:
    """Synthesis agent that combines all agent outputs into final meal plan."""
    req = state["meal_request"]
    dietary_preference = req["dietary_preference"]
    duration = req["duration"]
    
    prompt_t = (
        "Create a comprehensive {duration} meal plan for {dietary_preference} diet.\n\n"
        "Inputs:\nRecipes: {research}\nNutrition: {nutrition}\nSchedule: {planning}\nGroceries: {shopping}\n"
        "Provide a detailed, actionable meal plan with recipes, schedule, and shopping list."
    )
    vars_ = {
        "duration": duration,
        "dietary_preference": dietary_preference,
        "research": (state.get("research") or "")[:400],
        "nutrition": (state.get("nutrition") or "")[:400],
        "planning": (state.get("planning") or "")[:400],
        "shopping": (state.get("shopping") or "")[:400],
    }
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = meal_llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_meal_graph():
    """Build the meal planning StateGraph with parallel agent execution."""
    g = StateGraph(MealPlanState)
    g.add_node("research_node", research_agent)
    g.add_node("nutrition_node", nutrition_agent)
    g.add_node("planning_node", planning_agent)
    g.add_node("shopping_node", shopping_agent)
    g.add_node("synthesis_node", meal_synthesis_agent)

    # Run research, nutrition, planning, and shopping agents in parallel
    g.add_edge(START, "research_node")
    g.add_edge(START, "nutrition_node")
    g.add_edge(START, "planning_node")
    g.add_edge(START, "shopping_node")
    
    # All four agents feed into the synthesis agent
    g.add_edge("research_node", "synthesis_node")
    g.add_edge("nutrition_node", "synthesis_node")
    g.add_edge("planning_node", "synthesis_node")
    g.add_edge("shopping_node", "synthesis_node")
    
    g.add_edge("synthesis_node", END)

    return g.compile()


def plan_meals(request: MealPlanRequest) -> MealPlanResponse:
    """Main function to process meal planning requests."""
    graph = build_meal_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "meal_request": request.model_dump(),
        "tool_calls": [],
    }
    
    # Execute the graph
    result = graph.invoke(state)
    
    return MealPlanResponse(
        result=result.get("final", ""),
        grocery_list=result.get("shopping", ""),
        nutritional_info=result.get("nutrition", ""),
        tool_calls=result.get("tool_calls", [])
    )