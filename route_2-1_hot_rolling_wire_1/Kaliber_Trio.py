mean_roll_radius_1_upper = (328e-3 + 324e-3) / 2 / 2
mean_roll_radius_1_lower = (324e-3 + 320e-3) / 2 / 2
mean_roll_radius_2_upper = (299e-3 + 297e-3) / 2 / 2
mean_roll_radius_2_lower = (297e-3 + 295e-3) / 2 / 2
mean_roll_radius_3_upper = (280e-3 + 278e-3) / 2 / 2
mean_roll_radius_3_lower = (278e-3 + 276e-3) / 2 / 2

roll_gap_1_upper = 3e-3
roll_gap_1_lower = 3e-3
roll_gap_2_upper = 0.5e-3
roll_gap_2_lower = 1e-3
roll_gap_3_upper = 0.75e-3
roll_gap_3_lower = 0.5e-3


pr.RollPass(
        label="Raute I",
        roll=pr.Roll(
            groove=pr.DiamondGroove(
                usable_width=76.55e-3,
                tip_depth=22.1e-3,
                r1=12e-3,
                r2=8e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat II",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=52.7e-3,
                tip_depth=25.95e-3,
                r1=8e-3,
                r2=6e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Raute III",
        roll=pr.Roll(
            groove=pr.DiamondGroove(
                usable_width=58.3e-3,
                tip_depth=16.85e-3,
                r1=7e-3,
                r2=8e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat IV",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=40.74e-3,
                tip_depth=20.05e-3,
                r1=7e-3,
                r2=5e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Oval V",
        roll=pr.Roll(
            groove=pr.CircularOvalGroove(
                depth=7.25e-3,
                r1=6e-3,
                r2=44.5e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat VI",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=29.64e-3,
                tip_depth=14.625e-3,
                r1=6e-3,
                r2=4e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Oval VII",
        roll=pr.Roll(
            groove=pr.CircularOvalGroove(
                depth=5.05e-3,
                r1=7e-3,
                r2=33e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat VIII",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=21.54e-3,
                tip_depth=10.6e-3,
                r1=5e-3,
                r2=3e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Oval IX",
        roll=pr.Roll(
            groove=pr.CircularOvalGroove(
                depth=4.43e-3,
                r1=6e-3,
                r2=25.5e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat X",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=16.22e-3,
                tip_depth=8.11e-3,
                r1=2e-3,
                r2=1e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Fertigrund Xa",
        roll=pr.Roll(
            groove=pr.RoundGroove(
                r1=2e-3,
                r2=15.8e-3 / 2,
                depth=7.65e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Oval XI",
        roll=pr.Roll(
            groove=pr.CircularOvalGroove(
                depth=3.53e-3,
                r1=6e-3,
                r2=17e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Fertigrund XIIa",
        roll=pr.Roll(
            groove=pr.RoundGroove(
                r1=2e-3,
                r2=11.8e-3 / 2,
                depth=5.65e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat XII",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=13.58e-3,
                tip_depth=6.70e-3,
                r1=4e-3,
                r2=2e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Oval XIII",
        roll=pr.Roll(
            groove=pr.CircularOvalGroove(
                depth=2.7e-3,
                r1=5e-3,
                r2=13.2e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat XIV",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=10.73e-3,
                tip_depth=5.37e-3,
                r1=3.6e-3,
                r2=1.6e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Fertigrund XIVa",
        roll=pr.Roll(
            groove=pr.RoundGroove(
                r1=1.5e-3,
                r2=5e-3,
                depth=4.75e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Oval XV",
        roll=pr.Roll(
            groove=pr.CircularOvalGroove(
                depth=2.16e-3,
                r1=4e-3,
                r2=12.15e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Fertigrund XVIa",
        roll=pr.Roll(
            groove=pr.RoundGroove(
                r1=1.5e-3,
                r2=3.9e-3,
                depth=3.65e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Quadrat XVI",
        roll=pr.Roll(
            groove=pr.SquareGroove(
                usable_width=8.31e-3,
                tip_depth=4.16e-3,
                r1=3e-3,
                r2=0.62e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Oval XVIIa",
        roll=pr.Roll(
            groove=pr.CircularOvalGroove(
                depth=2.1e-3,
                r1=3e-3,
                r2=7.12e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),


pr.RollPass(
        label="Fertigrund XVIIIa",
        roll=pr.Roll(
            groove=pr.RoundGroove(
                r1=1.5e-3,
                r2=3.3e-3,
                depth=3.05e-3
            ),
            nominal_radius=,

        ),
        gap=,
        velocity=1,
        coulomb_friction_coefficient=0.4,
    ),