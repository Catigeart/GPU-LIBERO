"""Fail-closed field plans for the original LIBERO action-boundary predicates.

Plans change only the fields downloaded before Python task logic. They neither
evaluate nor memoize a predicate. Unknown callbacks keep the original full goal
schema, and every boundary checks the actual bound methods and task graph again.
This module is CPU-only and never imports a simulator or the GPU runtime.
"""
import ast
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import inspect
import sys
import textwrap


FULL_GOAL_FIELDS = ("qpos", "body_xpos", "body_xquat", "site_xpos",
                    "site_xmat", "contact", "ncon")
_PROBLEMS = {
    "Libero_Tabletop_Manipulation": "libero_tabletop_manipulation",
    "Libero_Kitchen_Tabletop_Manipulation": "libero_kitchen_tabletop_manipulation",
    "Libero_Living_Room_Tabletop_Manipulation": "libero_living_room_tabletop_manipulation",
    "Libero_Study_Tabletop_Manipulation": "libero_study_tabletop_manipulation",
    "Libero_Coffee_Table_Manipulation": "libero_coffee_table_manipulation",
    "Libero_Floor_Manipulation": "libero_floor_manipulation",
}
_BASE = "libero.libero.envs."
_OBJECT_MODULES = tuple(_BASE + "objects." + name for name in
                        ("google_scanned_objects", "hope_objects",
                         "turbosquid_objects", "articulated_objects"))
_PREDICATES = {"on": "On", "in": "In", "open": "Open", "close": "Close",
               "turnon": "TurnOn", "turnoff": "TurnOff"}

# AST digests of the CPU-reference methods whose dynamic read sets are encoded
# below. A changed/custom implementation is not admitted by its name alone.
_SOURCE_DIGESTS = {
    ('libero.libero.envs.bddl_base_domain', 'BDDLBaseDomain', 'reward'): '70bb86071f37557b48b1c3228922c00c078bd638ab622ee077cb5d66d5ffc396',
    ('libero.libero.envs.bddl_base_domain', 'BDDLBaseDomain', '_check_success'): 'cdbf15bb696e624d9745b5324e734225c2386f5e7b114786bd60e07f4458fc41',
    ('libero.libero.envs.bddl_base_domain', 'BDDLBaseDomain', '_post_action'): 'e5417248a92ab6a2be56811f8ffcc4831e1c7c81891231030fdd4893a6856632',
    ('libero.libero.envs.bddl_base_domain', 'BDDLBaseDomain', '_post_process'): '5c986098aa7da3eec045983402ebb614e78be6a8e165c0cd5248124f2fa632e3',
    ('libero.libero.envs.bddl_base_domain', 'BDDLBaseDomain', 'get_object'): 'a2de4358642bb74f5ecc1f03033c81eeed105395e0ee921c99971c281ad65ed2',
    ('libero.libero.envs.object_states.base_object_states', 'BaseObjectState', 'check_contact'): '7f00a696dba2e69e71f9cb11cae00656404ad67cd27de1bc71d5e59122628e73',
    ('libero.libero.envs.object_states.base_object_states', 'BaseObjectState', 'check_contain'): '73450626c0dcb9e0da89de834bc8a522af79d03c1897ffe3870c66f5b09ee18a',
    ('libero.libero.envs.object_states.base_object_states', 'BaseObjectState', 'is_open'): '43384fb7c35d8ef7136821c12edde920cd4ebb71ecec1fd3287c6d1da816e1d0',
    ('libero.libero.envs.object_states.base_object_states', 'BaseObjectState', 'is_close'): '58403ae952d02b829ffe65790f210f80a5e48f3ab289f766489de789f592e782',
    ('libero.libero.envs.object_states.base_object_states', 'BaseObjectState', 'check_ontop'): '8ac002323ebb2493f0d36fb6ea75795f2a2297766521de608e9637f53cf7f068',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'check_contact'): '1191610e3f6b29d557e223791f59b0ff0877f4d171435d9a7c0a57f7df198bc6',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'check_contain'): 'b70de3aecd51d2b37a2dd5cd3e355165fbc312c7a15696034eef035e6ce81437',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'check_ontop'): '2cc67812e3316f44e7e9a0ef9bcc3fa59128b9a78f67e41be5d1ec44fd39fadf',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'is_open'): 'b94b0155d4685d4b7dcc1d5e908a067750cd661b24f5467860ba032b6b4bb728',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'is_close'): '94232f53550fa53a37136c2cd3ee65873b140a3c5f32fdcd849591a8f181cf25',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'turn_on'): '578c1cd8b27b8a89377bfb906806dbe1f31ffd4e3bd6c1a793fa22f2b8fc7d3d',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'turn_off'): '84466c05aa89b4ecaf2ff58a8108d3c355b29cf3625562e09dcca909dbf9bb8d',
    ('libero.libero.envs.object_states.base_object_states', 'ObjectState', 'update_state'): '3cbc4af69ea4adc71cbbb31740b26700bcb02380455aa7ee9b2f5e54f00f5c9c',
    ('libero.libero.envs.object_states.base_object_states', 'SiteObjectState', 'check_contain'): '0941510f8c2ff796196cdf802d3fbe07316ffbb0c75308fb85ddb9008f91f8eb',
    ('libero.libero.envs.object_states.base_object_states', 'SiteObjectState', 'check_contact'): '0897b4e0c83b8b207a4e29f7e4acbcb4d0351b4a5bab4ee220effc0bf8c03ef4',
    ('libero.libero.envs.object_states.base_object_states', 'SiteObjectState', 'check_ontop'): '84e46fc67708b9f5df447aee0ff26f0c9de5e9f9d542a2ab6dea6bbb2de08419',
    ('libero.libero.envs.object_states.base_object_states', 'SiteObjectState', 'is_open'): '43dfad5a72b5328e6e7ee7da6eb51ca2dc1d95357865c6125deec966742fceb8',
    ('libero.libero.envs.object_states.base_object_states', 'SiteObjectState', 'is_close'): '312d767ad8e1dcd2ffa08b203263590064178f3bbdb0e3f40980198e32bcdd00',
    ('libero.libero.envs.predicates.base_predicates', 'Expression', '__call__'): '6933f23c0e90aff696619f297e1044021e6e1eb038cfb9e6791b650199c216c1',
    ('libero.libero.envs.predicates.base_predicates', 'UnaryAtomic', '__call__'): 'adef0b59a8a7b1a584761daf94a7d6f4711bd3a6c1b98d8feeb25f09fb6a52e5',
    ('libero.libero.envs.predicates.base_predicates', 'BinaryAtomic', '__call__'): '408ebba4925e28c9d3cab14011149115140653447346f65489d4998889a4e9e4',
    ('libero.libero.envs.predicates.base_predicates', 'MultiarayAtomic', '__call__'): 'ae72dc2e6c6e47e17483240c25d02067a84b7749322f92195409368ce4d28e0d',
    ('libero.libero.envs.predicates.base_predicates', 'TruePredicateFn', '__call__'): '9de7baa54a5d77c8dbcc62c2d5d5ab8b0767900e0c0d2ca591fda49f8c980b49',
    ('libero.libero.envs.predicates.base_predicates', 'FalsePredicateFn', '__call__'): '7d5912547a5576e5710f318a40b68a54ed7bfa24a1c2d38ccecd7cdc8dc5ce02',
    ('libero.libero.envs.predicates.base_predicates', 'InContactPredicateFn', '__call__'): 'b0999b48ebeb8312a9dd1b33cd690675e920fe391d17146cecfb2346439f92db',
    ('libero.libero.envs.predicates.base_predicates', 'In', '__call__'): 'e16894818b1682712f8a8ca05601e2318faf4ade543b8aa5c6725f09276724c8',
    ('libero.libero.envs.predicates.base_predicates', 'On', '__call__'): '281775bf9cc45ed748052b0b4b9eed71ea8f6f96e1654c980ae5324c5c8eeeab',
    ('libero.libero.envs.predicates.base_predicates', 'Up', '__call__'): '3b7d0b4ee148213ecda16e6dd08d28b73bacdcabcc660ce7d20e3076b7d24f45',
    ('libero.libero.envs.predicates.base_predicates', 'Stack', '__call__'): '724e69b6ffdfda78c0efd43397e439f621cae696cc3bed222d4d18030d8438ac',
    ('libero.libero.envs.predicates.base_predicates', 'PrintJointState', '__call__'): '394b9c752ba2a036d10242ad065a4e048bd883ab8957ea075bbdefc43ed5f994',
    ('libero.libero.envs.predicates.base_predicates', 'Open', '__call__'): '35d501f02f3b6abbb186f488e23a8d164334da225927099d6e5a857890de0075',
    ('libero.libero.envs.predicates.base_predicates', 'Close', '__call__'): 'ee3342530de96ed5aeefa1cb37af559b513e4bc6f7331406b9a0e37e28a5a8b2',
    ('libero.libero.envs.predicates.base_predicates', 'TurnOn', '__call__'): '098fe841875e3e756cc5077ce11799452ca844c5c9e5f9d6f4017355c6a50292',
    ('libero.libero.envs.predicates.base_predicates', 'TurnOff', '__call__'): 'f6d6273438115ceae2f42aa641bfe05a1c3ccf0627f7464afde9e46562f0142d',
    ('libero.libero.envs.predicates', '', 'eval_predicate_fn'): 'cfca765651a4caa11c2d6fd4c977d1c93b85e90564de48b0d59af06439a4ea21',
    ('libero.libero.envs.objects.articulated_objects', 'ArticulatedObject', 'is_open'): '5157fbc96025560ebb05c134d7906f031d5872b873ae7b152d40c7527f75128f',
    ('libero.libero.envs.objects.articulated_objects', 'ArticulatedObject', 'is_close'): '987678bbcee8dd355c2dc31a07a547f85ea43ac2d30470545822c939cd69baf3',
    ('libero.libero.envs.objects.articulated_objects', 'Microwave', 'is_open'): '6e1bd213df1f59f58f6350afad06bc906e5089ac56ca7a77be3a203ea6df3b5d',
    ('libero.libero.envs.objects.articulated_objects', 'Microwave', 'is_close'): 'd2c7383a895417479eac3fbf44aed47feb7213de069098e386eacc96ba39e890',
    ('libero.libero.envs.objects.articulated_objects', 'ShortCabinet', 'is_open'): '2cbb949a948fafbdc4629c2f10680a4ad285b79f31915e81b0eafc929f5684f3',
    ('libero.libero.envs.objects.articulated_objects', 'ShortCabinet', 'is_close'): '7b769655df0208abf85f8fd05ce37fa4f5300f27d5a2c90a848f19e4fd206c52',
    ('libero.libero.envs.objects.articulated_objects', 'ShortFridge', 'is_open'): '2cbb949a948fafbdc4629c2f10680a4ad285b79f31915e81b0eafc929f5684f3',
    ('libero.libero.envs.objects.articulated_objects', 'ShortFridge', 'is_close'): '7b769655df0208abf85f8fd05ce37fa4f5300f27d5a2c90a848f19e4fd206c52',
    ('libero.libero.envs.objects.articulated_objects', 'WoodenCabinet', 'is_open'): '6e1bd213df1f59f58f6350afad06bc906e5089ac56ca7a77be3a203ea6df3b5d',
    ('libero.libero.envs.objects.articulated_objects', 'WoodenCabinet', 'is_close'): 'd2c7383a895417479eac3fbf44aed47feb7213de069098e386eacc96ba39e890',
    ('libero.libero.envs.objects.articulated_objects', 'WhiteCabinet', 'is_open'): '6e1bd213df1f59f58f6350afad06bc906e5089ac56ca7a77be3a203ea6df3b5d',
    ('libero.libero.envs.objects.articulated_objects', 'WhiteCabinet', 'is_close'): 'd2c7383a895417479eac3fbf44aed47feb7213de069098e386eacc96ba39e890',
    ('libero.libero.envs.objects.articulated_objects', 'FlatStove', 'turn_on'): 'fe9680c8bf61188bd049f93b33924a78677afb03a32e3a6c801e379cc33ce717',
    ('libero.libero.envs.objects.articulated_objects', 'FlatStove', 'turn_off'): '61f17f8f3cbc5833b60b1d9ff07b2d0bfc1c7900010888f75a20ae996198b5c9',
    ('libero.libero.envs.objects.site_object', 'SiteObject', 'in_box'): '56db48fda7b00b181f68f512ce5b39d95b2e606e163673c704e4907b41d85d3d',
    ('libero.libero.envs.objects.site_object', 'SiteObject', 'under'): '1868536c7c18a66ee6ad10e66e5356bf91345c4893db1d3e85351decfc9f96e4',
    ('libero.libero.envs.objects.target_zones', 'TargetZone', 'in_box'): '56db48fda7b00b181f68f512ce5b39d95b2e606e163673c704e4907b41d85d3d',
    ('libero.libero.envs.problems.libero_coffee_table_manipulation', 'Libero_Coffee_Table_Manipulation', '_check_success'): '7e68f19dbf933bd3d6ec5f5ef82434747180184d573feb4bfdde1a35b98cc7b7',
    ('libero.libero.envs.problems.libero_coffee_table_manipulation', 'Libero_Coffee_Table_Manipulation', '_eval_predicate'): '45c4cb92df963b4d3bd5be6fdd1af323f39668642cf5cd02f1858084b3efad7c',
    ('libero.libero.envs.problems.libero_coffee_table_manipulation', 'Libero_Coffee_Table_Manipulation', '_post_process'): 'c8801d040bcb49b3b0d061c67b360aa8a399f3a687e28bc47c2ab4e4de7531c4',
    ('libero.libero.envs.problems.libero_coffee_table_manipulation', 'Libero_Coffee_Table_Manipulation', 'set_visualization'): '954e0757498a241be3f9f8326cd7bad11c19c651bc066256938d86d781005cc7',
    ('libero.libero.envs.problems.libero_floor_manipulation', 'Libero_Floor_Manipulation', '_check_success'): '7e68f19dbf933bd3d6ec5f5ef82434747180184d573feb4bfdde1a35b98cc7b7',
    ('libero.libero.envs.problems.libero_floor_manipulation', 'Libero_Floor_Manipulation', '_eval_predicate'): '45c4cb92df963b4d3bd5be6fdd1af323f39668642cf5cd02f1858084b3efad7c',
    ('libero.libero.envs.problems.libero_floor_manipulation', 'Libero_Floor_Manipulation', '_post_process'): 'c8801d040bcb49b3b0d061c67b360aa8a399f3a687e28bc47c2ab4e4de7531c4',
    ('libero.libero.envs.problems.libero_floor_manipulation', 'Libero_Floor_Manipulation', 'set_visualization'): '954e0757498a241be3f9f8326cd7bad11c19c651bc066256938d86d781005cc7',
    ('libero.libero.envs.problems.libero_kitchen_tabletop_manipulation', 'Libero_Kitchen_Tabletop_Manipulation', '_check_success'): '7e68f19dbf933bd3d6ec5f5ef82434747180184d573feb4bfdde1a35b98cc7b7',
    ('libero.libero.envs.problems.libero_kitchen_tabletop_manipulation', 'Libero_Kitchen_Tabletop_Manipulation', '_eval_predicate'): '45c4cb92df963b4d3bd5be6fdd1af323f39668642cf5cd02f1858084b3efad7c',
    ('libero.libero.envs.problems.libero_kitchen_tabletop_manipulation', 'Libero_Kitchen_Tabletop_Manipulation', '_post_process'): 'c8801d040bcb49b3b0d061c67b360aa8a399f3a687e28bc47c2ab4e4de7531c4',
    ('libero.libero.envs.problems.libero_kitchen_tabletop_manipulation', 'Libero_Kitchen_Tabletop_Manipulation', 'set_visualization'): '954e0757498a241be3f9f8326cd7bad11c19c651bc066256938d86d781005cc7',
    ('libero.libero.envs.problems.libero_living_room_tabletop_manipulation', 'Libero_Living_Room_Tabletop_Manipulation', '_check_success'): '7e68f19dbf933bd3d6ec5f5ef82434747180184d573feb4bfdde1a35b98cc7b7',
    ('libero.libero.envs.problems.libero_living_room_tabletop_manipulation', 'Libero_Living_Room_Tabletop_Manipulation', '_eval_predicate'): '45c4cb92df963b4d3bd5be6fdd1af323f39668642cf5cd02f1858084b3efad7c',
    ('libero.libero.envs.problems.libero_living_room_tabletop_manipulation', 'Libero_Living_Room_Tabletop_Manipulation', '_post_process'): 'c8801d040bcb49b3b0d061c67b360aa8a399f3a687e28bc47c2ab4e4de7531c4',
    ('libero.libero.envs.problems.libero_living_room_tabletop_manipulation', 'Libero_Living_Room_Tabletop_Manipulation', 'set_visualization'): '954e0757498a241be3f9f8326cd7bad11c19c651bc066256938d86d781005cc7',
    ('libero.libero.envs.problems.libero_study_tabletop_manipulation', 'Libero_Study_Tabletop_Manipulation', '_check_success'): '7e68f19dbf933bd3d6ec5f5ef82434747180184d573feb4bfdde1a35b98cc7b7',
    ('libero.libero.envs.problems.libero_study_tabletop_manipulation', 'Libero_Study_Tabletop_Manipulation', '_eval_predicate'): '45c4cb92df963b4d3bd5be6fdd1af323f39668642cf5cd02f1858084b3efad7c',
    ('libero.libero.envs.problems.libero_study_tabletop_manipulation', 'Libero_Study_Tabletop_Manipulation', '_post_process'): 'c8801d040bcb49b3b0d061c67b360aa8a399f3a687e28bc47c2ab4e4de7531c4',
    ('libero.libero.envs.problems.libero_study_tabletop_manipulation', 'Libero_Study_Tabletop_Manipulation', 'set_visualization'): '954e0757498a241be3f9f8326cd7bad11c19c651bc066256938d86d781005cc7',
    ('libero.libero.envs.problems.libero_tabletop_manipulation', 'Libero_Tabletop_Manipulation', '_check_success'): '7e68f19dbf933bd3d6ec5f5ef82434747180184d573feb4bfdde1a35b98cc7b7',
    ('libero.libero.envs.problems.libero_tabletop_manipulation', 'Libero_Tabletop_Manipulation', '_eval_predicate'): '45c4cb92df963b4d3bd5be6fdd1af323f39668642cf5cd02f1858084b3efad7c',
    ('libero.libero.envs.problems.libero_tabletop_manipulation', 'Libero_Tabletop_Manipulation', '_post_process'): 'c8801d040bcb49b3b0d061c67b360aa8a399f3a687e28bc47c2ab4e4de7531c4',
    ('libero.libero.envs.problems.libero_tabletop_manipulation', 'Libero_Tabletop_Manipulation', 'set_visualization'): '954e0757498a241be3f9f8326cd7bad11c19c651bc066256938d86d781005cc7',
    ('robosuite.environments.base', 'MujocoEnv', '_post_action'): 'b98de5e6515e9203162bded024c74170ddd0405c0468d861e927fc8166d16648',
    ('robosuite.environments.base', 'MujocoEnv', 'reward'): '5ebd747179423848e1d7e47f3fb014c96759346979b921063839f86b92d92b58',
    ('robosuite.environments.base', 'MujocoEnv', 'check_contact'): 'fac8a60ec84a24f358a6ef283fce05f343c16bf5274e8ac866c339b1039920f4',
    ('robosuite.environments.base', 'MujocoEnv', '_check_success'): '59ae2b9b828cb2b7df58f4d2ce4dc5f6dab3531b478c6ebe96007be19694ffa9',
    ('robosuite.models.base', 'MujocoModel', 'correct_naming'): 'ef0e8a116dfc47ede4ca2f72036f6651e8cf80eef1f1abe7a66f9733f0b2b71b',
    ('robosuite.models.base', 'MujocoModel', 'exclude_from_prefixing'): 'd6dc37e0f91973bdef606d62e8068ed2994166099eb6811f2e1676bfd98b246c',
    ('robosuite.models.base', 'MujocoModel', 'naming_prefix'): '8f9d15b5b5f4f641036591ef105108418b4e42fc838db1ce157af0374be5f56d',
    ('robosuite.models.base', 'MujocoModel', 'joints'): '8034102c1f4ff1d05185ff4d92cac17f1fc7e4026b8b3cd91fb04f14f6f9bcdf',
    ('robosuite.models.base', 'MujocoModel', 'contact_geoms'): '032d69abd297cb1bcb887f8601d31555a40e62fb64b8687b18b5b7557b9b785e',
    ('robosuite.models.base', 'MujocoXMLModel', 'exclude_from_prefixing'): '7d17d5b1e3741b8379ec0ca5c9b65b40975ad2ce1814d806225b561b6c4f8c92',
    ('robosuite.models.base', 'MujocoXMLModel', 'naming_prefix'): 'e0214324d0538ee4c250edce007fc409a50c20a18309c59563ed3e5acc804bea',
    ('robosuite.models.base', 'MujocoXMLModel', 'joints'): '2a0fcb77f10c7b03bca7ee48bf1bb680568ca0e737c1424efdf1acb5f3cc0487',
    ('robosuite.models.base', 'MujocoXMLModel', 'contact_geoms'): 'a62c4a822262d579a3afb834073163b3e4af23b23533541a460b821821f1c87b',
    ('robosuite.models.objects.objects', 'MujocoObject', 'exclude_from_prefixing'): 'd6dc37e0f91973bdef606d62e8068ed2994166099eb6811f2e1676bfd98b246c',
    ('robosuite.models.objects.objects', 'MujocoObject', 'naming_prefix'): '95d58faf62cc3414e89d4f83203f90c6fec456cd7ad402c65c2dc20e440b4795',
    ('robosuite.models.objects.objects', 'MujocoObject', 'joints'): '2a0fcb77f10c7b03bca7ee48bf1bb680568ca0e737c1424efdf1acb5f3cc0487',
    ('robosuite.models.objects.objects', 'MujocoObject', 'contact_geoms'): 'a62c4a822262d579a3afb834073163b3e4af23b23533541a460b821821f1c87b',
    ('robosuite.models.objects.objects', 'MujocoXMLObject', 'exclude_from_prefixing'): '7d17d5b1e3741b8379ec0ca5c9b65b40975ad2ce1814d806225b561b6c4f8c92',
    ('robosuite.models.objects.objects', 'MujocoGeneratedObject', 'exclude_from_prefixing'): 'fa9542a59f3ad8d30a5455419ac1b15db4676a71b1e59a9e24bdbf1877aa40d1',
}


def _loaded(module, name):
    # register_problem intentionally returns None: the original class lives in
    # TASK_MAPPING, while the decorated module attribute is None.
    if name in _PROBLEMS and module == _BASE + "problems." + _PROBLEMS[name]:
        registry = getattr(sys.modules.get(_BASE + "bddl_base_domain"), "TASK_MAPPING", {})
        value = registry.get(name.lower())
        return value if isinstance(value, type) and value.__module__ == module else None
    return getattr(sys.modules.get(module), name, None)


def _function(value):
    return getattr(value, "__func__", value)


def _source_matches(function, key):
    return _source_matches_code(function.__code__, key)


@lru_cache(maxsize=None)
def _source_matches_code(code, key):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(code)))
        node = tree.body[0]
        node.decorator_list = []
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        return digest == _SOURCE_DIGESTS.get(key)
    except (OSError, TypeError, SyntaxError, AttributeError):
        return False


def _freeze(value):
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    if type(value) is dict:
        return tuple((key, _freeze(item)) for key, item in value.items())
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("non-builtin task descriptor")


def _method_token(value):
    function = _function(value)
    return (function, getattr(value, "__self__", None), function.__code__,
            function.__defaults__, _freeze(function.__kwdefaults__))


class _Proof:
    def __init__(self):
        self.checks = []

    def attr(self, owner, name, *, method=False):
        value = getattr(owner, name)
        token = _method_token(value) if method else _freeze(value)
        self.checks.append(("method" if method else "attr", owner, name, token))
        return value

    def item(self, mapping, key):
        value = mapping[key]
        self.checks.append(("item", mapping, key, value))
        return value

    def canonical_class(self, module, class_name, cls):
        if class_name in _PROBLEMS and module == _BASE + "problems." + _PROBLEMS[class_name]:
            owner = sys.modules[_BASE + "bddl_base_domain"]
            self.checks.append(("identity_attr", owner, "TASK_MAPPING", owner.TASK_MAPPING))
            self.item(owner.TASK_MAPPING, class_name.lower())
        else:
            self.checks.append(("identity_attr", sys.modules[module], class_name, cls))

    def function(self, function, module, name):
        if not _source_matches(function, (module, "", name)):
            raise ValueError("custom dispatcher: " + name)
        self.checks.append(("function", function, "", _method_token(function)))

    def method(self, owner, name, module, class_name):
        cls = _loaded(module, class_name)
        if cls is None:
            raise ValueError("canonical method owner is not loaded")
        expected = getattr(cls, name)
        value = self.attr(owner, name, method=True)
        if (_function(value) is not _function(expected) or
                not _source_matches(_function(expected), (module, class_name, name))):
            raise ValueError("custom method: " + name)
        # Class replacement is also checked, even when an old bound method was
        # retained on the instance before replacement.
        self.canonical_class(module, class_name, cls)
        self.checks.append(("identity_attr", cls, name, expected))
        return value

    def property(self, owner, name, module, class_name):
        cls = _loaded(module, class_name)
        expected = inspect.getattr_static(cls, name) if cls is not None else None
        if (not isinstance(expected, property) or
                inspect.getattr_static(type(owner), name) is not expected or
                not _source_matches(expected.fget, (module, class_name, name))):
            raise ValueError("custom object property: " + name)
        self.checks.append(("property", type(owner), name, (expected, _method_token(expected.fget))))


def _check(check):
    kind, owner, name, expected = check
    if kind == "item":
        return owner[name] is expected
    if kind == "identity_attr":
        return getattr(owner, name) is expected
    if kind == "property":
        prop, token = expected
        return inspect.getattr_static(owner, name) is prop and _method_token(prop.fget) == token
    if kind == "function":
        return _method_token(owner) == expected
    if kind == "goal":
        return _freeze(getattr(owner, name)["goal_state"]) == expected
    if kind == "sequence_identity":
        current = getattr(owner, name)
        return len(current) == len(expected) and all(a is b for a, b in zip(current, expected))
    if kind == "mapping_keys":
        return tuple(owner) == expected
    if kind == "mro":
        return owner.__mro__ == expected
    value = getattr(owner, name)
    if kind == "method":
        return _method_token(value) == expected
    return _freeze(value) == expected


@dataclass(frozen=True)
class GoalReadPlan:
    fields: tuple = FULL_GOAL_FIELDS
    qualified: bool = False
    reason: str = "unqualified task callbacks"
    _owner: object = field(default=None, repr=False, compare=False)
    _checks: tuple = field(default=(), repr=False, compare=False)

    def fields_for(self, shell):
        """Select the schema before any reward/post-process side effects run."""
        if self.qualified and shell is self._owner:
            try:
                if all(_check(check) for check in self._checks):
                    return self.fields
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        return FULL_GOAL_FIELDS

    @classmethod
    def from_shell(cls, shell):
        try:
            fields, proof = _plan(shell)
            ordered = tuple(name for name in FULL_GOAL_FIELDS if name in fields)
            return cls(ordered, True, "original LIBERO predicates and trackers", shell, tuple(proof.checks))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            return cls(reason=str(error), _owner=shell)


def _plan(shell):
    proof = _Proof()
    name = type(shell).__name__
    module = _BASE + "problems." + _PROBLEMS[name]
    if type(shell) is not _loaded(module, name):
        raise ValueError("custom task class")
    proof.canonical_class(module, name, type(shell))
    proof.checks.append(("identity_attr", shell, "__class__", type(shell)))
    proof.checks.append(("mro", type(shell), "", type(shell).__mro__))
    for method in ("_check_success", "_eval_predicate", "_post_process", "set_visualization"):
        proof.method(shell, method, module, name)
    for method in ("reward", "_post_action", "get_object"):
        proof.method(shell, method, _BASE + "bddl_base_domain", "BDDLBaseDomain")
    # The original task _post_action delegates to this method before its own
    # post-processing. A custom parent implementation needs the full schema.
    env_cls = _loaded("robosuite.environments.base", "MujocoEnv")
    if env_cls is None or not _source_matches(env_cls._post_action,
                    ("robosuite.environments.base", "MujocoEnv", "_post_action")):
        raise ValueError("custom parent post_action")
    base = _loaded(_BASE + "bddl_base_domain", "BDDLBaseDomain")
    if getattr(super(base, shell), "_post_action", None).__func__ is not env_cls._post_action:
        raise ValueError("custom post_action superclass route")
    proof.checks.append(("identity_attr", env_cls, "_post_action", env_cls._post_action))
    proof.checks.append(("function", env_cls._post_action, "", _method_token(env_cls._post_action)))
    proof.method(shell, "check_contact", "robosuite.environments.base", "MujocoEnv")
    proof.method(base, "_post_process", _BASE + "bddl_base_domain", "BDDLBaseDomain")
    for attr in ("objects_dict", "fixtures_dict", "object_sites_dict", "object_states_dict"):
        if type(getattr(shell, attr)) is not dict:
            raise ValueError("custom task graph mapping")
        proof.checks.append(("identity_attr", shell, attr, getattr(shell, attr)))
        proof.checks.append(("mapping_keys", getattr(shell, attr), "", tuple(getattr(shell, attr))))
    goals = shell.parsed_problem
    if type(goals) is not dict:
        raise ValueError("custom parsed problem")
    proof.checks.append(("goal", shell, "parsed_problem", _freeze(goals["goal_state"])))
    fields = set()
    state_module = _BASE + "object_states.base_object_states"
    object_state = _loaded(state_module, "ObjectState")
    site_state = _loaded(state_module, "SiteObjectState")
    registry = _loaded(_BASE + "predicates", "VALIDATE_PREDICATE_FN_DICT")
    if object_state is None or site_state is None or type(registry) is not dict:
        raise ValueError("original predicate/state registry is unavailable")
    predicate_module = _BASE + "predicates"
    dispatcher = _loaded(predicate_module, "eval_predicate_fn")
    proof.function(dispatcher, predicate_module, "eval_predicate_fn")
    evaluator_globals = _function(shell._eval_predicate).__globals__
    if (evaluator_globals.get("eval_predicate_fn") is not dispatcher or
            dispatcher.__globals__.get("VALIDATE_PREDICATE_FN_DICT") is not registry):
        raise ValueError("custom predicate dispatcher globals")
    proof.item(evaluator_globals, "eval_predicate_fn")
    proof.item(dispatcher.__globals__, "VALIDATE_PREDICATE_FN_DICT")
    proof.checks.append(("identity_attr", sys.modules[predicate_module], "eval_predicate_fn", dispatcher))
    proof.checks.append(("identity_attr", sys.modules[predicate_module], "VALIDATE_PREDICATE_FN_DICT", registry))

    def object_for(object_name):
        obj = shell.get_object(object_name)
        if obj is None:
            return None
        for mapping in (shell.fixtures_dict, shell.objects_dict, shell.object_sites_dict):
            if object_name in mapping:
                proof.item(mapping, object_name)
                break
        typ = type(obj)
        if (typ.__module__ not in _OBJECT_MODULES or
                typ is not _loaded(typ.__module__, typ.__name__) or
                typ.__getattribute__ is not object.__getattribute__):
            raise ValueError("custom object class")
        proof.checks.append(("identity_attr", sys.modules[typ.__module__], typ.__name__, typ))
        proof.checks.append(("identity_attr", obj, "__class__", typ))
        proof.checks.append(("identity_attr", typ, "__getattribute__", object.__getattribute__))
        proof.attr(obj, "name")
        # These original properties only transform static XML names. A custom
        # property/correct_naming could instead read arbitrary simulator fields.
        for attr in ("joints", "contact_geoms"):
            proof.property(obj, attr, "robosuite.models.objects.objects", "MujocoObject")
        proof.property(obj, "naming_prefix", "robosuite.models.objects.objects", "MujocoObject")
        proof.method(obj, "correct_naming", "robosuite.models.base", "MujocoModel")
        proof.method(obj, "exclude_from_prefixing", "robosuite.models.objects.objects", "MujocoXMLObject")
        return obj

    def state_for(object_name):
        state = proof.item(shell.object_states_dict, object_name)
        if type(state) not in (object_state, site_state) or state.env is not shell:
            raise ValueError("custom object state")
        proof.checks.append(("identity_attr", state, "__class__", type(state)))
        proof.checks.append(("identity_attr", state, "env", shell))
        proof.attr(state, "object_name")
        if state.object_name != object_name:
            raise ValueError("object state name changed")
        state_name = "ObjectState" if type(state) is object_state else "SiteObjectState"
        proof.checks.append(("identity_attr", sys.modules[state_module], state_name, type(state)))
        if type(state) is object_state:
            proof.attr(state, "is_fixture")
            object_for(object_name)
        else:
            proof.attr(state, "parent_name")
            site = proof.item(shell.object_sites_dict, object_name)
            site_type = type(site)
            allowed = ((_BASE + "objects.site_object", "SiteObject"),
                       (_BASE + "objects.target_zones", "TargetZone"))
            if not any(site_type is _loaded(m, c) for m, c in allowed):
                raise ValueError("custom site object")
            proof.checks.append(("identity_attr", site, "__class__", site_type))
            proof.attr(site, "parent_name")
            proof.attr(site, "joints")
        return state, state_name

    for atom in goals["goal_state"]:
        if type(atom) not in (list, tuple) or len(atom) not in (2, 3):
            raise ValueError("unsupported goal expression")
        predicate = atom[0]
        predicate_name = _PREDICATES[predicate]
        fn = proof.item(registry, predicate)
        canonical = _loaded(_BASE + "predicates.base_predicates", predicate_name)
        if type(fn) is not canonical:
            raise ValueError("custom predicate registry entry")
        proof.checks.append(("identity_attr", fn, "__class__", canonical))
        proof.method(fn, "__call__", _BASE + "predicates.base_predicates", predicate_name)
        states = [state_for(object_name) for object_name in atom[1:]]
        if predicate in ("open", "close", "turnon", "turnoff"):
            if len(states) != 1 or (predicate in ("turnon", "turnoff") and states[0][1] != "ObjectState"):
                raise ValueError("unsupported articulation predicate")
            state, state_name = states[0]
            method = {"open": "is_open", "close": "is_close", "turnon": "turn_on", "turnoff": "turn_off"}[predicate]
            proof.method(state, method, state_module, state_name)
            obj = object_for(atom[1] if state_name == "ObjectState" else state.parent_name)
            if obj is None:
                raise ValueError("articulation parent is absent")
            proof.method(obj, method, _BASE + "objects.articulated_objects", type(obj).__name__)
            fields.add("qpos")
        else:
            if len(states) != 2 or states[0][1] != "ObjectState":
                raise ValueError("unsupported relation operands")
            target, target_name = states[1]
            proof.method(target, "check_ontop" if predicate == "on" else "check_contain", state_module, target_name)
            fields.add("body_xpos")
            if target_name == "SiteObjectState":
                site = shell.object_sites_dict[atom[2]]
                site_class = "TargetZone" if type(site).__name__ == "TargetZone" else "SiteObject"
                site_module = _BASE + ("objects.target_zones" if site_class == "TargetZone" else "objects.site_object")
                proof.method(site, "under" if predicate == "on" else "in_box",
                             _BASE + "objects.site_object" if predicate == "on" else site_module,
                             "SiteObject" if predicate == "on" else site_class)
                fields.update(("site_xpos", "site_xmat"))
                if predicate == "in":
                    proof.method(target, "check_contact", state_module, target_name)
                elif object_for(target.parent_name) is not None:
                    fields.update(("contact", "ncon"))
            else:
                # Current catalog only has On(object,object). Generic In may
                # call an object-specific in_box implementation; keep it full.
                if predicate != "on":
                    raise ValueError("object containment is not qualified")
                proof.method(target, "check_contact", state_module, target_name)
                fields.update(("contact", "ncon"))

    trackers = shell.tracking_object_states_change
    if type(trackers) not in (list, tuple):
        raise ValueError("custom tracker collection")
    proof.checks.append(("sequence_identity", shell, "tracking_object_states_change", tuple(trackers)))
    for tracker in trackers:
        if type(tracker) is not object_state or tracker.env is not shell:
            raise ValueError("custom object tracker")
        proof.checks.append(("identity_attr", tracker, "__class__", object_state))
        proof.checks.append(("identity_attr", tracker, "env", shell))
        proof.attr(tracker, "object_name")
        proof.method(tracker, "update_state", state_module, "ObjectState")
        proof.method(tracker, "turn_on", state_module, "ObjectState")
        proof.attr(tracker, "has_turnon_affordance")
        if tracker.has_turnon_affordance is not True:
            raise ValueError("unknown tracker affordance")
        obj = object_for(tracker.object_name)
        if type(obj) is not _loaded(_BASE + "objects.articulated_objects", "FlatStove"):
            raise ValueError("unqualified visual tracker")
        proof.method(obj, "turn_on", _BASE + "objects.articulated_objects", "FlatStove")
        fields.add("qpos")
    return fields, proof
