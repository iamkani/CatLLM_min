# Role-grounded few-shot exemplars for CatLLM_min
# Keep answers brand-agnostic, concise (<150 words), and neutral.

ROLE_KEYS = {
    "association_analyst": "Association Analyst",
    "buyer_feeder": "Buyer / Feeder",
    "genetic_advisor": "Genetic Advisor",
    "independent_rancher": "Independent Rancher",
    "root": "Root",
}

def canonical_role(name):
    if not name:
        return "Association Analyst"
    norm = name.strip().lower().replace("/", "_").replace(" ", "_")
    return ROLE_KEYS.get(norm, "Association Analyst")

FEWSHOTS = {
    "Association Analyst": [
        {"role": "user", "content": "What is the Beef Checkoff and can it be used for lobbying?"},
        {"role": "assistant", "content": "The Beef Checkoff is a mandatory $1-per-head assessment used for promotion, research, and education. By law, funds cannot be used to influence policy or lobby. Typical activities include consumer education, nutrition research, and market development."},
        {"role": "user", "content": "How do USDA grades differ from brand labels?"},
        {"role": "assistant", "content": "USDA grades (Prime, Choice, Select) are official quality grades based mainly on marbling and maturity. Brands are marketing programs that may set additional specs, but they don't replace USDA grades. Grades are standardized across plants; brand criteria vary by program."},
        {"role": "user", "content": "Where can I find official beef production statistics?"},
        {"role": "assistant", "content": "Use federal datasets for authoritative statistics, e.g., USDA NASS for herd counts and production, USDA AMS for market reports, and ERS for trade and consumption. Cite publication dates when available."},
        {"role": "user", "content": "What does 'grass-fed' vs. 'organic' generally mean?"},
        {"role": "assistant", "content": "'Grass-fed' typically refers to diet, while 'organic' refers to a certified system covering feed, land, and animal health inputs. Requirements and labeling claims are defined by regulators; check current rules and program documents for specifics."},
    ],
    "Buyer / Feeder": [
        {"role": "user", "content": "How much freezer space do I need for a half beef?"},
        {"role": "assistant", "content": "Plan roughly 30-35 lb of packaged beef per cubic foot. A typical half yields about 200-220 lb packaged, so budget ~7-8 cubic feet. Vacuum-sealed cuts extend storage and reduce freezer burn."},
        {"role": "user", "content": "What's the difference between backgrounding and finishing?"},
        {"role": "assistant", "content": "Backgrounding is the growth phase after weaning to build frame and health at moderate gains. Finishing targets higher energy diets for marbling and slaughter weights. Programs vary by region and goals."},
        {"role": "user", "content": "How do I read a cut sheet for custom processing?"},
        {"role": "assistant", "content": "Cut sheets map primals to retail cuts and thickness, grind percentages, and packaging. Confirm steak thickness, roast size, bone-in vs. boneless, grind fat %, and special items (brisket, soup bones). Ask about aging windows and packaging (vacuum vs. paper)."},
        {"role": "user", "content": "What temperatures are safe for cooking and storage?"},
        {"role": "assistant", "content": "Refrigerate at <=40F (4C); freeze at 0F (-18C). Cook ground beef to 160F (71C); steaks/roasts to at least 145F (63C) with rest time, per common food-safety guidance. Use a thermometer; local advisories may differ."},
    ],
    "Genetic Advisor": [
        {"role": "user", "content": "Is calving ease the same as selecting for low birth weight?"},
        {"role": "assistant", "content": "Birth weight influences dystocia but calving ease also reflects calf shape, dam size/parity, and management. Use calving-ease EPDs or indexes rather than BW alone, and match sires to heifers vs. cows appropriately."},
        {"role": "user", "content": "What's the value of heterosis in crossbreeding?"},
        {"role": "assistant", "content": "Heterosis (hybrid vigor) can improve traits like fertility, longevity, and disease resistance. Structured crossbreeding captures both maternal and terminal advantages; design systems with herd goals and markets in mind."},
        {"role": "user", "content": "How do EPDs and genomic-enhanced EPDs differ?"},
        {"role": "assistant", "content": "EPDs estimate genetic merit from pedigree and performance data. Genomic-enhanced EPDs include DNA information, increasing accuracy, especially in young animals with limited records. Compare across the same evaluation system."},
        {"role": "user", "content": "What should I prioritize in beef-on-dairy matings?"},
        {"role": "assistant", "content": "Prioritize calving ease for heifers, then growth and carcass traits that fit target grids. Select for polled status when desired; manage with clear breeding objectives and calf marketing plan."},
    ],
    "Independent Rancher": [
        {"role": "user", "content": "Are antibiotics routinely used in beef cattle?"},
        {"role": "assistant", "content": "Use is oversight-driven and subject to withdrawal times before harvest. Programs emphasize judicious use, veterinary guidance, and record-keeping. If buyers require specific protocols, confirm those in advance."},
        {"role": "user", "content": "Grass-finished vs. grain-finished - what are the trade-offs?"},
        {"role": "assistant", "content": "Grass-finishing can align with forage resources and certain flavor profiles; grain-finishing typically shortens time to grade and boosts marbling. Choose based on resources, markets, and risk tolerance; both require sound nutrition and health programs."},
        {"role": "user", "content": "What basic prep is needed to sell calves?"},
        {"role": "assistant", "content": "Standardize ID, maintain vaccination and treatment records, deworm as needed, and consider preconditioning/backgrounding. Good handling, uniform groups, and clear disclosures support better market outcomes."},
        {"role": "user", "content": "Do I need special facilities to start?"},
        {"role": "assistant", "content": "Start with safe handling areas (gates, alley, head catch), clean water, mineral access, and adequate fencing. Plan shade/windbreaks based on climate and a simple record system for inventory and treatments."},
    ],
}

def get_fewshots_for_role(role_name):
    return FEWSHOTS.get(canonical_role(role_name), [])
