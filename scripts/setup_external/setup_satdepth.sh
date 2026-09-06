#!/bin/bash

# clone satdepth
mkdir -p ../../external/satdepth
git clone https://github.com/rahuldeshmukh43/satdepth.git ../../external/satdepth

# satdepth's own code does absolute self-imports (e.g. `import satdepth.src.utils...`),
# and the clone is named "satdepth" itself (not a differently-named package living inside
# it), so it's external/ (the clone's *parent*) that needs to be on PYTHONPATH directly for
# those self-imports to resolve -- scripts/epimask/*.sh add external/ to PYTHONPATH for this
# reason. epimask's own code imports it as epimask.external.satdepth.src...
