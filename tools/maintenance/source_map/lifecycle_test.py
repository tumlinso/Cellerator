#!/usr/bin/env python3
"""Exercise source-map generation and staleness gates in a temporary Git repo."""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

TOOLS=Path(__file__).resolve().parent

def run(command:list[str],cwd:Path,ok:bool=True)->str:
    result=subprocess.run(command,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if ok and result.returncode:
        raise RuntimeError(f"{command} failed ({result.returncode}):\n{result.stdout}")
    if not ok and result.returncode==0:
        raise RuntimeError(f"{command} unexpectedly passed:\n{result.stdout}")
    return result.stdout

def main()->None:
    with tempfile.TemporaryDirectory(prefix="cellerator-source-map-") as temp:
        root=Path(temp); repo=root/"repo"; tools=repo/"tools"/"maintenance"/"source_map"
        out="docs/development/source_map"
        (repo/"src"/"compute").mkdir(parents=True)
        tools.mkdir(parents=True)
        (repo/".gitignore").write_text("__pycache__/\n*.pyc\nbuild/\n")
        (repo/"README.md").write_text("# fixture\n")
        (repo/"src"/"compute"/"model.py").write_text("value = 1\n")
        for name in ("generate.py","validate.py","common.py","lifecycle_test.py"):
            shutil.copy2(TOOLS/name,tools/name)
        run(["git","init","-q"],repo)
        run(["git","config","user.email","fixture@example.invalid"],repo)
        run(["git","config","user.name","Source Map Fixture"],repo)
        run(["git","add","."],repo); run(["git","commit","-qm","baseline"],repo)
        def generate():
            return run(["python","-B",str(tools/"generate.py"),"--repo",str(repo),"--output",out],repo)
        def validate(ok:bool=True):
            return run(["python","-B",str(tools/"validate.py"),"--repo",str(repo),"--output",out],repo,ok)
        def audit_migrations(ok:bool=True):
            return run(["python","-B",str(tools/"validate.py"),"--repo",str(repo),"--output",out,
                        "--migration-manifest","planning/coherence/source-migrations.json"],repo,ok)
        generate(); run(["git","add","."],repo); run(["git","commit","-qm","generated source map"],repo); validate()

        # Advancing HEAD by committing generated outputs must not invalidate itself.
        (repo/"README.md").write_text("# source edit\n")
        validate(ok=False); run(["git","add","README.md"],repo); generate(); validate()
        run(["git","add","."],repo); run(["git","commit","-qm","source edit and refreshed map"],repo); validate()

        # An unknown path remains rejected even after generation.
        (repo/"mystery.bin").write_bytes(b"opaque source\n")
        run(["git","add","mystery.bin"],repo); run(["git","commit","-qm","unknown path"],repo)
        validate(ok=False); generate(); validate(ok=False)
        generator=tools/"generate.py"; source=generator.read_text()
        fallback="    # A concrete gap, intentionally not a blanket claim.\n"
        explicit=("    if p == 'mystery.bin':\n"
                  "        return ('fixture source','fixture owner','fixture operation family','retain_fixture_source','source_present_unqualified','unknown','unknown','preserve original path and blob')\n")
        if fallback not in source:
            raise RuntimeError("could not locate explicit route insertion point")
        generator.write_text(source.replace(fallback,explicit+fallback))
        validate(ok=False); run(["git","add",str(generator.relative_to(repo))],repo); generate(); validate()
        run(["git","add","."],repo); run(["git","commit","-qm","explicit route and refreshed map"],repo); validate()

        # Source moves change both the path set and bound source identity.
        run(["git","mv","src/compute/model.py","src/compute/model-renamed.py"],repo)
        run(["git","commit","-qm","source move"],repo); validate(ok=False)
        generate(); validate()
        run(["git","add","."],repo); run(["git","commit","-qm","move and refreshed map"],repo); validate()

        # Any generated output edit is detected and can be reproduced.
        rules=repo/out/"routing_rules.json"
        rules.write_text(rules.read_text()+"\n")
        validate(ok=False); generate(); validate()

        # Generic source-map validation needs no migration manifest; explicit audit catches ignored intended sources.
        (repo/"include"/"Cellerator"/"compiler"/"build").mkdir(parents=True)
        destination=repo/"include"/"Cellerator"/"compiler"/"build"/"intended.hh"
        destination.write_text("// ignored source\n")
        pathspec_decoy=repo/"include"/"Cellerator"/"compiler"/"build"/"literalb.hh"
        pathspec_decoy.write_text("// pathspec decoy\n")
        (repo/"planning"/"coherence").mkdir(parents=True)
        migration_manifest=repo/"planning"/"coherence"/"source-migrations.json"
        migration_manifest.write_text(json.dumps({
            "path_migrations":[
                {"new_path":"include/Cellerator/compiler/build/intended.hh","target_path_exists":True},
                {"new_path":"include/Cellerator/compiler/build/literal[bracket].hh","target_path_exists":True},
            ],
            "source_extractions":[],"compatibility_facades":[],
        })+"\n")
        run(["git","add","planning/coherence/source-migrations.json"],repo)
        run(["git","add","-f","include/Cellerator/compiler/build/literalb.hh"],repo)
        run(["git","commit","-qm","record migration destination"],repo)
        validate(ok=False); generate(); validate()
        missing_destination=audit_migrations(ok=False)
        if "intended migration destination is not tracked: include/Cellerator/compiler/build/intended.hh" not in missing_destination:
            raise RuntimeError("migration audit did not identify the ignored intended destination")
        literal_missing="intended migration destination is not tracked: include/Cellerator/compiler/build/literal[bracket].hh"
        if literal_missing not in missing_destination:
            raise RuntimeError("migration audit treated a pathspec match as an exact tracked filename")
        run(["git","add","-f","include/Cellerator/compiler/build/intended.hh"],repo)
        run(["git","commit","-qm","track intended migration destination"],repo)
        validate(ok=False); generate(); validate()
        still_missing=audit_migrations(ok=False)
        if literal_missing not in still_missing:
            raise RuntimeError("pathspec decoy incorrectly satisfied an absent literal destination")
        literal_destination=repo/"include"/"Cellerator"/"compiler"/"build"/"literal[bracket].hh"
        literal_destination.write_text("// exact literal source\n")
        run(["git","add","-f","--",":(literal)include/Cellerator/compiler/build/literal[bracket].hh"],repo)
        run(["git","commit","-qm","track literal bracket destination"],repo)
        validate(ok=False); generate(); validate(); audit_migrations()
        print(json.dumps({
            "ok":True,"lifecycle":"generate -> stage/commit -> validate",
            "cases":[
                "committed generated map does not self-invalidate and HEAD is provenance only",
                "explicit migration audit catches ignored intended destinations while ordinary validation remains manifest-agnostic",
                "source edit invalidates the map until regeneration",
                "new unknown route fails both before and after regeneration",
                "adding an explicit route changes the bound rules and regenerating restores validation",
                "source move invalidates path coverage until regeneration",
                "generated-output corruption is detected and regenerated",
                "generic source-map validation works without a migration manifest; explicit audit catches ignored intended destinations and passes once tracked",
                "migration auditing compares literal tracked filenames; bracket pathspec metacharacters do not accept a decoy",
            ],
            "final_head":run(["git","rev-parse","HEAD"],repo).strip(),
        },indent=2))

if __name__=="__main__":
    main()
