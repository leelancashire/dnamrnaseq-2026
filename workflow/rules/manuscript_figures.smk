"""Manuscript figure generation rules.

Generates all figures bound to the paper. All stubs until analysis is complete.
"""


rule render_atlas_figure:
    """Render the headline trajectory-atlas figure (stub)."""
    input:
        atlas   = "analysis/latest/trajectory_atlas.csv",
        archs   = "analysis/latest/archetypes.csv",
        annot   = "analysis/latest/recovery_axis_annotation.csv",
    output:
        fig     = "manuscript/figures/fig1_trajectory_atlas.png",
    shell:
        "echo 'TODO: implement trajectory atlas figure rendering'"


rule all_figures:
    """Generate all manuscript figures (stub)."""
    input:
        rules.render_atlas_figure.output.fig,
    shell:
        "echo 'All figures generated.'"
