"""
Complete user journey integration tests that simulate the real user experience.

Tests the full A-to-B journey:
1. First-time user onboarding (from scratch)
2. Subsequent app opens (returning user experience)

Uses pytest-order to ensure proper test sequence and shared state.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# Session-scoped fixture to maintain test environment across the user journey
@pytest.fixture(scope="session")
def user_journey_environment():
    """
    Session-scoped fixture that creates and maintains test environment
    across the complete user journey tests.
    """
    home_dir = tempfile.mkdtemp(prefix="fuzzy_journey_")
    
    # Create shell history for realistic onboarding
    zsh_history = Path(home_dir) / ".zsh_history" 
    zsh_history.write_text(": 1609459200:0;echo hello world\n: 1609459210:0;ls -la\n: 1609459220:0;git status\n: 1609459230:0;npm install\n: 1609459240:0;docker ps\n")
    
    bash_history = Path(home_dir) / ".bash_history"
    bash_history.write_text(": 0:0;echo hello\n: 0:0;ls -la\n: 0:0;git commit -m 'fix'\n")
    
    # Ensure fuzzyshell directory exists but is empty (forces onboarding)
    fuzzyshell_dir = Path(home_dir) / ".fuzzyshell"
    fuzzyshell_dir.mkdir(parents=True, exist_ok=True)
    
    # Store environment data that needs to persist between tests
    env_data = {
        'home_dir': home_dir,
        'command_count': None,
        'has_ann_index': None
    }
    
    yield env_data
    
    # Cleanup after all tests complete
    if Path(home_dir).exists():
        shutil.rmtree(home_dir, ignore_errors=True)
        print(f"🧹 Session cleanup: {home_dir}")


class SharedTestHelpers:
    """Shared helper methods for performing search tests across user journey"""
    
    @staticmethod
    def perform_search_test(fuzzyshell_instance, query="echo", expected_min_results=1):
        """
        Shared helper to test search functionality consistently.
        
        Args:
            fuzzyshell_instance: FuzzyShell instance to test
            query: Search query to test
            expected_min_results: Minimum number of results expected
            
        Returns:
            tuple: (results, success_bool)
        """
        try:
            results = fuzzyshell_instance.search(query, top_k=5)
            success = len(results) >= expected_min_results
            return results, success
        except Exception as e:
            print(f"Search test failed: {e}")
            return [], False
    
    @staticmethod
    def verify_database_state(fuzzyshell_instance, expected_min_commands=1):
        """
        Verify the database has the expected state.
        
        Args:
            fuzzyshell_instance: FuzzyShell instance to check
            expected_min_commands: Minimum number of commands expected
            
        Returns:
            dict: Database state information
        """
        try:
            command_count = fuzzyshell_instance.get_indexed_count()
            system_info = fuzzyshell_instance.get_system_info()
            db_info = system_info.get('database', {})
            
            return {
                'command_count': command_count,
                'actual_command_count': db_info.get('actual_command_count', 0),
                'has_ann_index': db_info.get('has_ann_index', False),
                'success': command_count >= expected_min_commands
            }
        except Exception as e:
            print(f"Database state check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def verify_search_works(fuzzyshell_instance, test_scenario=""):
        """
        Shared assertion helper that performs a comprehensive search verification.
        This is NOT a test itself, but a helper function both tests can call.
        
        Args:
            fuzzyshell_instance: FuzzyShell instance to test
            test_scenario: Description of test scenario (for error messages)
        
        Raises:
            AssertionError: If any search functionality fails
        """
        scenario_prefix = f"[{test_scenario}] " if test_scenario else ""
        
        # Test 1: Basic text search
        results, success = SharedTestHelpers.perform_search_test(fuzzyshell_instance, "echo", 1)
        assert success, f"{scenario_prefix}Basic search should work, got {len(results)} results for 'echo'"
        
        # Test 2: Different command search  
        git_results, git_success = SharedTestHelpers.perform_search_test(fuzzyshell_instance, "git", 1)
        assert git_success, f"{scenario_prefix}Git search should work, got {len(git_results)} results for 'git'"
        
        # Test 3: Semantic search (if ANN is available)
        try:
            semantic_results, semantic_success = SharedTestHelpers.perform_search_test(fuzzyshell_instance, "list files", 1)
            # Note: Don't assert semantic search - it's optional and depends on ANN availability
            if semantic_success:
                print(f"{scenario_prefix}✅ Semantic search working: {len(semantic_results)} results for 'list files'")
            else:
                print(f"{scenario_prefix}ℹ️  Semantic search not available or no results")
        except Exception:
            # Semantic search failures are not critical
            print(f"{scenario_prefix}ℹ️  Semantic search test skipped")
        
        print(f"{scenario_prefix}✅ All critical search functionality verified")


@pytest.mark.order(1)
def test_first_time_onboarding_complete_journey(user_journey_environment):
    """
    Test the complete first-time user onboarding experience.
    
    This test runs FIRST and simulates a brand new user:
    1. Shows onboarding screens 
    2. Downloads models
    3. Ingests command history
    4. Enables search functionality
    5. Stores environment for subsequent test
    """
    
    home_dir = user_journey_environment['home_dir']
    
    with patch.dict(os.environ, {'HOME': home_dir}):
        # Add src to path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        
        # Test onboarding screen components are available
        from fuzzyshell.tui.simple_onboarding import check_needs_setup
        from fuzzyshell.tui.screens.onboarding import StartupManager, OnboardingStage
        
        # 1. VERIFY: Onboarding is needed
        assert check_needs_setup(), "Should need setup in fresh environment"
        
        # 2. VERIFY: Onboarding screens show expected stages to user
        manager = StartupManager()
        
        # Test the key stages users will see
        expected_user_stages = [
            (OnboardingStage.CHECKING_MODELS, "Checking Models"),
            (OnboardingStage.DOWNLOADING_EMBEDDING, "Downloading Embedding Model (1/2)"),
            (OnboardingStage.DOWNLOADING_DESCRIPTION, "Downloading Description Model (2/2)"), 
            (OnboardingStage.INITIALIZING, "Initializing FuzzyShell"),
            (OnboardingStage.COMPLETE, "Ready")
        ]
        
        stage_names_shown = []
        for stage, expected_name in expected_user_stages:
            manager.update_progress(stage, 25, f"Testing {expected_name}")
            info = manager.get_stage_info()
            stage_names_shown.append(info['stage_name'])
            
        # Assert all expected stage names are shown to user
        for _, expected_name in expected_user_stages:
            assert expected_name in stage_names_shown, f"User should see '{expected_name}' stage"
            
        # 3. EXECUTE: Complete onboarding (models + ingestion)
        # Use stock model for integration tests (allows download)
        os.environ['FUZZYSHELL_MODEL'] = 'stock-minilm-l6'
        
        from fuzzyshell import FuzzyShell
        
        fuzzyshell = FuzzyShell()
        
        # Verify models are available after initialization
        assert fuzzyshell._model is not None or fuzzyshell.model is not None, \
            "Models should be available after onboarding"
        
        # Perform ingestion
        count = fuzzyshell.ingest_history(use_tui=False, no_random=True)
        count = count or 0
        assert count > 0, f"Should ingest commands from history, got {count}"
        
        # 4. TEST: Search functionality works after onboarding
        SharedTestHelpers.verify_search_works(fuzzyshell, "First-time onboarding")
        
        # 5. VERIFY: Database state is good
        db_state = SharedTestHelpers.verify_database_state(fuzzyshell, expected_min_commands=count)
        assert db_state['success'], f"Database should have commands after onboarding: {db_state}"
        
        # Store final state for subsequent test
        user_journey_environment['command_count'] = db_state['command_count']
        user_journey_environment['has_ann_index'] = db_state['has_ann_index']
        
        print(f"✅ First-time onboarding complete: {count} commands, search working")


@pytest.mark.order(2) 
def test_subsequent_app_open_returning_user(user_journey_environment):
    """
    Test the returning user experience.
    
    This test runs SECOND and simulates returning to the app:
    1. No onboarding screens (already setup)
    2. Database persisted correctly
    3. Search still works
    """
    
    home_dir = user_journey_environment['home_dir']
    if not user_journey_environment['command_count']:
        pytest.skip("First-time onboarding test didn't run or failed")
    
    with patch.dict(os.environ, {'HOME': home_dir}):
        # Add src to path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        
        # 1. VERIFY: Database persists (returning user has data)
        # Note: Models may be downloaded on-demand, so we check for database persistence
        db_path = os.path.join(home_dir, ".fuzzyshell", "fuzzyshell.db")
        assert os.path.exists(db_path), "Database should persist for returning user"
        
        # 2. EXECUTE: Create new FuzzyShell instance (simulates app restart)
        # Use stock model for integration tests (allows download)
        os.environ['FUZZYSHELL_MODEL'] = 'stock-minilm-l6'
        
        from fuzzyshell import FuzzyShell
        
        fuzzyshell = FuzzyShell()
        
        # 3. VERIFY: Database persisted correctly
        expected_count = user_journey_environment['command_count']
        db_state = SharedTestHelpers.verify_database_state(fuzzyshell, expected_min_commands=expected_count)
        
        assert db_state['success'], f"Database should persist for returning user: {db_state}"
        assert db_state['command_count'] == expected_count, \
            f"Command count should match previous session: expected {expected_count}, got {db_state['command_count']}"
        
        # 4. TEST: Search still works for returning user (comprehensive verification)
        SharedTestHelpers.verify_search_works(fuzzyshell, "Returning user")
            
        print(f"✅ Returning user experience verified: {db_state['command_count']} commands, search working")


