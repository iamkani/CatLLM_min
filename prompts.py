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
        {"role": "assistant", "content": (
            "The Beef Checkoff is a mandatory $1-per-head assessment used for promotion, research, "
            "and education under the Beef Promotion and Research Act of 1985. Funds cannot be used "
            "to lobby or influence policy. See: "
            "[USDA AMS – Beef Checkoff overview](https://www.ams.usda.gov/rules-regulations/research-promotion/beef), "
            "[GAO summary of checkoff restrictions](https://www.gao.gov/assets/gao-18-55r.pdf)."
        )},
        {"role": "user", "content": "How do USDA grades differ from brand labels?"},
        {"role": "assistant", "content": (
            "USDA grades (Prime, Choice, Select) are official AMS quality grades based mainly on marbling and maturity. "
            "Brands are private programs with their own specs; they do not replace USDA grades. See "
            "[AMS beef grades & standards](https://www.ams.usda.gov/grades-standards/carcass-beef-grades-and-standards) "
            "and [USDA explainer](https://www.usda.gov/media/blog/2013/01/28/whats-your-beef-prime-choice-or-select)."
        )},
        {"role": "user", "content": "Where can I find official beef production statistics?"},
        {"role": "assistant", "content": (
            "Use federal datasets: [USDA NASS Quick Stats](https://www.nass.usda.gov/Quick_Stats/) for inventory/production, "
            "[USDA AMS Market News/MPR Data Mart](https://mpr.datamart.ams.usda.gov/) for price/volume, and "
            "[USDA ERS – Cattle & Beef](https://www.ers.usda.gov/topics/animal-products/cattle-beef/sector-at-a-glance/) "
            "for trade/consumption analysis."
        )},
        {"role": "user", "content": "What does 'grass-fed' vs. 'organic' generally mean?"},
        {"role": "assistant", "content": (
            "'Grass-fed' is an animal-raising label claim verified by FSIS based on documentation; "
            "'organic' is certification under USDA's National Organic Program. See "
            "[FSIS claims guidance](https://www.fsis.usda.gov/inspection/compliance-guidance/labeling/claims-guidance) "
            "and [USDA NOP overview](https://www.ams.usda.gov/about-ams/programs-offices/national-organic-program)."
        )},
    ],

    "Buyer / Feeder": [
        {"role": "user", "content": "How much freezer space do I need for a half beef?"},
        {"role": "assistant", "content": (
            "Plan roughly 30-35 lb of packaged beef per cubic foot. A typical half yields about 200-220 lb packaged, "
            "so budget around 7-8 cubic feet. Vacuum-sealed cuts extend storage and reduce freezer burn."
        )},
        {"role": "user", "content": "What's the difference between backgrounding and finishing?"},
        {"role": "assistant", "content": (
            "Backgrounding is the post-weaning growth phase to build frame and health at moderate gains. "
            "Finishing uses higher-energy diets targeting marbling and slaughter weights; specifics vary by region and goals."
        )},
        {"role": "user", "content": "How do I read a cut sheet for custom processing?"},
        {"role": "assistant", "content": (
            "Cut sheets map primals to retail cuts, thickness, grind percentages, and packaging. "
            "Confirm steak thickness, roast size, bone-in vs boneless, grind fat %, and special items (brisket, soup bones). "
            "Ask about aging windows and packaging (vacuum vs paper)."
        )},
        {"role": "user", "content": "What temperatures are safe for cooking and storage?"},
        {"role": "assistant", "content": (
            "Refrigerate at <=40F (4C); freeze at 0F (-18C). Cook ground beef to 160F (71C); "
            "steaks/roasts to 145F (63C) with rest time. See "
            "[FoodSafety.gov temperature chart](https://www.foodsafety.gov/food-safety-charts/safe-minimum-internal-temperatures) "
            "and [USDA FSIS guidance](https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/how-temperatures-affect-food)."
        )},
    ],

    "Genetic Advisor": [
        {"role": "user", "content": "Is calving ease the same as selecting for low birth weight?"},
        {"role": "assistant", "content": (
            "Birth weight influences dystocia, but calving ease also reflects calf shape, dam size/parity, and management. "
            "Use calving-ease EPDs or indexes rather than BW alone, and match sires to heifers vs cows appropriately."
        )},
        {"role": "user", "content": "What's the value of heterosis in crossbreeding?"},
        {"role": "assistant", "content": (
            "Heterosis (hybrid vigor) can improve fertility, longevity, and resilience. "
            "Structured crossbreeding captures maternal and terminal advantages aligned to herd goals and markets."
        )},
        {"role": "user", "content": "How do EPDs and genomic-enhanced EPDs differ?"},
        {"role": "assistant", "content": (
            "EPDs estimate genetic merit from pedigree and performance. Genomic-enhanced EPDs add DNA info, "
            "increasing accuracy, especially in young animals with limited records. Compare within the same evaluation system."
        )},
        {"role": "user", "content": "What should I prioritize in beef-on-dairy matings?"},
        {"role": "assistant", "content": (
            "Prioritize calving ease for heifers, then growth and carcass traits that fit target grids. "
            "Select for polled status when desired and align with a clear calf-marketing plan."
        )},
    ],

    "Independent Rancher": [
        {"role": "user", "content": "Are antibiotics routinely used in beef cattle?"},
        {"role": "assistant", "content": (
            "Use is oversight-driven with required withdrawal times before harvest. Programs emphasize judicious use, "
            "veterinary guidance, and records. Buyer-required protocols may apply; confirm in advance. "
            "See [FDA antimicrobial stewardship](https://www.fda.gov/animal-veterinary/antimicrobial-resistance/antimicrobial-stewardship-veterinary-settings) "
            "and [Beef Quality Assurance basics](https://www.bqa.org/programs/bqa-programs)."
        )},
        {"role": "user", "content": "Grass-finished vs. grain-finished - what are the trade-offs?"},
        {"role": "assistant", "content": (
            "Grass-finishing can fit forage resources and certain flavor profiles; grain-finishing typically shortens time to grade "
            "and boosts marbling. Choose based on resources, markets, and risk tolerance; both require sound nutrition and health programs."
        )},
        {"role": "user", "content": "What basic prep is needed to sell calves?"},
        {"role": "assistant", "content": (
            "Standardize ID, keep vaccination/treatment records, deworm as needed, and consider preconditioning/backgrounding. "
            "Uniform groups and clear disclosures support better market outcomes."
        )},
        {"role": "user", "content": "Do I need special facilities to start?"},
        {"role": "assistant", "content": (
            "Start with safe handling areas (gates, alley, head catch), clean water, mineral access, and solid fencing. "
            "Add shade/windbreaks as needed and a simple record system for inventory and treatments."
        )},
    ],
}

def get_fewshots_for_role(role_name):
    return FEWSHOTS.get(canonical_role(role_name), [])
