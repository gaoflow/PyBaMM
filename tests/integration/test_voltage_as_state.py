"""Integration tests for voltage as a state behavior.

Verifies that voltage is always an algebraic state in standard (non-basic)
models.
"""

import warnings

import numpy as np
import pytest

import pybamm


class TestVoltageAlwaysAState:
    """Voltage should always be an algebraic state variable."""

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_voltage_is_algebraic_state(self, model_cls):
        model = model_cls()
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" in algebraic_var_names
        assert isinstance(model.variables["Voltage [V]"], pybamm.Variable)

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_residuals_reference_variable(self, model_cls):
        """Algebraic residuals should contain the Variable, not the expression."""
        model = model_cls()
        for var, expr in model.algebraic.items():
            if var.name == "Voltage [V]":
                syms = [
                    s
                    for s in expr.pre_order()
                    if isinstance(s, pybamm.Variable) and s.name == "Voltage [V]"
                ]
                assert len(syms) > 0

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_voltage_expression_matches_state(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])

        v = sol["Voltage [V]"].entries
        v_expr = sol["Voltage expression [V]"].entries
        np.testing.assert_allclose(v, v_expr, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    )
    def test_voltage_false_excludes_algebraic_voltage(self, model_cls):
        """When voltage-as-a-state is false, Voltage [V] is not an algebraic variable."""
        model = model_cls(
            options={"voltage as a state": "false", "surface form": "false"}
        )
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" not in algebraic_var_names
        assert not isinstance(model.variables["Voltage [V]"], pybamm.Variable)

    def test_dfn_vaas_false_still_has_algebraic_states(self):
        """DFN with vaas=false + surface_form=false still has algebraic states
        (electrode/electrolyte potentials), so it is NOT a pure ODE model."""
        model = pybamm.lithium_ion.DFN(
            options={"voltage as a state": "false", "surface form": "false"}
        )
        assert len(model.algebraic) > 0
        algebraic_var_names = [var.name for var in model.algebraic.keys()]
        assert "Voltage [V]" not in algebraic_var_names

    def test_dfn_vaas_false_rejects_scipy(self):
        """DFN with vaas=false still has algebraic states and cannot use ScipySolver."""
        model = pybamm.lithium_ion.DFN(
            options={"voltage as a state": "false", "surface form": "false"}
        )
        sim = pybamm.Simulation(model, solver=pybamm.ScipySolver())
        with pytest.raises(pybamm.SolverError, match="Cannot use ODE solver"):
            sim.solve([0, 3600])

    @pytest.mark.parametrize(
        "operating_mode",
        ["explicit power", "explicit resistance"],
    )
    def test_vaas_false_rejects_explicit_power_resistance(self, operating_mode):
        """Explicit power/resistance modes require voltage as a state."""
        with pytest.raises(
            pybamm.OptionError,
            match=r"Cannot use.*operating mode.*'voltage as a state'.*'false'",
        ):
            pybamm.lithium_ion.SPM(
                options={
                    "voltage as a state": "false",
                    "surface form": "false",
                    "operating mode": operating_mode,
                }
            )

    def test_spm_defaults_to_algebraic_surface_form(self):
        model = pybamm.lithium_ion.SPM()
        assert model.options["surface form"] == "algebraic"

    def test_spme_defaults_to_algebraic_surface_form(self):
        model = pybamm.lithium_ion.SPMe()
        assert model.options["surface form"] == "algebraic"

    def test_dfn_defaults_to_false_surface_form(self):
        model = pybamm.lithium_ion.DFN()
        assert model.options["surface form"] == "false"

    def test_vaas_false_emits_deprecation_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pybamm.lithium_ion.SPM(
                options={"voltage as a state": "false", "surface form": "false"}
            )
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "deprecated" in str(dep_warnings[0].message)
            assert "SPM/SPMe" in str(dep_warnings[0].message)
            assert "DFN" in str(dep_warnings[0].message)

    def test_vaas_true_does_not_emit_deprecation(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pybamm.lithium_ion.SPM(options={"voltage as a state": "true"})
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


class TestBasicModelsVoltageExpression:
    """Basic models expose voltage as an expression, not a state."""

    @pytest.mark.parametrize(
        "model_cls",
        [pybamm.lithium_ion.BasicSPM, pybamm.lithium_ion.BasicDFN],
    )
    def test_basic_models_have_voltage_expression(self, model_cls):
        model = model_cls()
        assert "Voltage expression [V]" in model.variables
        assert "Voltage [V]" in model.variables
        assert not isinstance(model.variables["Voltage [V]"], pybamm.Variable)


class TestLegacyOdeBehavior:
    """Verify that voltage-as-a-state=false + surface_form=false produces
    a working ODE model for SPM/SPMe (but not DFN, which retains other
    algebraic states)."""

    LEGACY_OPTIONS = {"voltage as a state": "false", "surface form": "false"}

    @pytest.mark.parametrize(
        "model_cls",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
        ],
    )
    def test_legacy_solvable_by_scipy(self, model_cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = model_cls(options=self.LEGACY_OPTIONS)
        sim = pybamm.Simulation(model, solver=pybamm.ScipySolver())
        sol = sim.solve([0, 3600])
        assert sol is not None
        v = sol["Voltage [V]"].entries
        assert v[0] > v[-1]  # voltage decreases during discharge

    @pytest.mark.parametrize(
        "model_cls",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
        ],
    )
    def test_legacy_solvable_by_casadi_safe(self, model_cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = model_cls(options=self.LEGACY_OPTIONS)
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="safe"))
        sol = sim.solve([0, 3600])
        assert sol is not None

    @pytest.mark.parametrize(
        "model_cls",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
        ],
    )
    def test_legacy_solvable_by_casadi_fast(self, model_cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = model_cls(options=self.LEGACY_OPTIONS)
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="fast"))
        sol = sim.solve([0, 3600])
        assert sol is not None


class TestDefaultDaeBehavior:
    """Verify that default SPM/SPMe (VAAS + algebraic surface form) works
    with DAE-capable solvers."""

    @pytest.mark.parametrize(
        "model_cls",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
            pybamm.lithium_ion.DFN,
        ],
    )
    def test_default_solvable_by_idaklu(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())
        sol = sim.solve([0, 3600])
        assert sol is not None
        v = sol["Voltage [V]"].entries
        assert v[0] > v[-1]

    @pytest.mark.parametrize(
        "model_cls",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
            pybamm.lithium_ion.DFN,
        ],
    )
    def test_default_solvable_by_casadi_safe(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="safe"))
        sol = sim.solve([0, 3600])
        assert sol is not None

    @pytest.mark.parametrize(
        "model_cls",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
            pybamm.lithium_ion.DFN,
        ],
    )
    def test_default_rejects_scipy(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model, solver=pybamm.ScipySolver())
        with pytest.raises(pybamm.SolverError, match="Cannot use ODE solver"):
            sim.solve([0, 3600])

    @pytest.mark.parametrize(
        "model_cls",
        [
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
            pybamm.lithium_ion.DFN,
        ],
    )
    def test_voltage_state_matches_expression(self, model_cls):
        model = model_cls()
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())
        sol = sim.solve([0, 3600])
        v = sol["Voltage [V]"].entries
        v_expr = sol["Voltage expression [V]"].entries
        np.testing.assert_allclose(v, v_expr, rtol=1e-3, atol=1e-3)
