import { useCallback } from 'react';
import PropTypes from 'prop-types';
import { MOVE_CURSOR_MODES, MOVE_CURSOR_CONFIG } from '../config';

/**
 * MovesPanel - UI panel for the move annotation workflow
 * Allows users to define start holds and multiple finish holds for batch move creation
 */
function MovesPanel({ moveParams, holds }) {
  const {
    currentMove,
    resetCurrentMove,
    moveCursorMode,
    setMoveCursorMode,
    isCurrentMoveValid,
    addCurrentMoveToMoves,
    moves,
    removeMove,
    clearMoves,
    exportMoves
  } = moveParams;

  // Get hold label by ID
  const getHoldLabel = useCallback((holdId) => {
    if (holdId === null || holdId === undefined) return '—';
    const hold = holds.find(h => h.hold_id === holdId);
    return hold ? `#${hold.hold_id}` : `#${holdId}`;
  }, [holds]);

  // Handle export
  const handleExport = useCallback(() => {
    const data = exportMoves();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `moves_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [exportMoves]);

  return (
    <div className="moves-panel">
      <h3>Move Annotation</h3>

      {/* Cursor Mode Selection */}
      <div className="moves-section">
        <h4>Click Mode</h4>
        <div className="cursor-grid">
          {Object.values(MOVE_CURSOR_MODES).map(mode => (
            <button
              key={mode}
              onClick={() => setMoveCursorMode(mode)}
              className={`cursor-btn ${moveCursorMode === mode ? 'active' : ''}`}
              style={{ 
                '--cursor-color': MOVE_CURSOR_CONFIG[mode].color,
                borderColor: moveCursorMode === mode ? MOVE_CURSOR_CONFIG[mode].color : undefined
              }}
              title={MOVE_CURSOR_CONFIG[mode].description}
            >
              {MOVE_CURSOR_CONFIG[mode].label}
              {mode.includes('finish') && ' ⊕'}
            </button>
          ))}
        </div>
        <p className="moves-hint">
          {MOVE_CURSOR_CONFIG[moveCursorMode].description}
          {moveCursorMode.includes('finish') && ' — click multiple holds'}
        </p>
      </div>

      {/* Current Move Preview */}
      <div className="moves-section current-move-box">
        <h4>Current Move</h4>
        
        {/* Start Holds */}
        <div className="move-row">
          <div className="move-col">
            <span className="move-label" style={{ color: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.LH_START].color }}>
              LH Start:
            </span>
            <span className="move-value">{getHoldLabel(currentMove.lh_start)}</span>
          </div>
          <div className="move-col">
            <span className="move-label" style={{ color: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.RH_START].color }}>
              RH Start:
            </span>
            <span className="move-value">{getHoldLabel(currentMove.rh_start)}</span>
          </div>
        </div>

        {/* Finish Holds */}
        <div className="move-row">
          <div className="move-col">
            <span className="move-label" style={{ color: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.LH_FINISH].color }}>
              LH Finish:
            </span>
            <div className="hold-chips">
              {currentMove.lh_finish.length === 0 ? (
                <span className="empty-chips">none</span>
              ) : (
                currentMove.lh_finish.map(holdId => (
                  <span 
                    key={holdId}
                    className="hold-chip"
                    style={{ backgroundColor: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.LH_FINISH].color }}
                  >
                    #{holdId}
                  </span>
                ))
              )}
            </div>
          </div>
          <div className="move-col">
            <span className="move-label" style={{ color: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.RH_FINISH].color }}>
              RH Finish:
            </span>
            <div className="hold-chips">
              {currentMove.rh_finish.length === 0 ? (
                <span className="empty-chips">none</span>
              ) : (
                currentMove.rh_finish.map(holdId => (
                  <span 
                    key={holdId}
                    className="hold-chip"
                    style={{ backgroundColor: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.RH_FINISH].color }}
                  >
                    #{holdId}
                  </span>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Current Move Actions */}
        <div className="button-group">
          <button
            onClick={addCurrentMoveToMoves}
            disabled={!isCurrentMoveValid()}
            className="btn btn-save"
          >
            Submit Move
          </button>
          <button
            onClick={resetCurrentMove}
            className="btn btn-secondary"
          >
            Clear
          </button>
        </div>
      </div>

      <hr className="moves-divider" />

      {/* Moves List */}
      <div className="moves-section">
        <h4>Saved Moves ({moves.length})</h4>
        
        {moves.length === 0 ? (
          <p className="moves-empty">No moves saved yet</p>
        ) : (
          <div className="moves-list">
            {moves.map((move, index) => (
              <div key={index} className={`move-item ${index % 2 === 0 ? 'even' : 'odd'}`}>
                <div className="move-item-info">
                  <span className="move-item-num">#{index + 1}</span>
                  <span className="move-item-detail">
                    <span style={{ color: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.LH_START].color }}>
                      L:{move.lh_start ?? '—'}→[{move.lh_finish.join(',') || '—'}]
                    </span>
                    {' '}
                    <span style={{ color: MOVE_CURSOR_CONFIG[MOVE_CURSOR_MODES.RH_START].color }}>
                      R:{move.rh_start ?? '—'}→[{move.rh_finish.join(',') || '—'}]
                    </span>
                  </span>
                </div>
                <button
                  onClick={() => removeMove(index)}
                  className="btn btn-danger btn-tiny"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Export Actions */}
      <div className="button-group">
        <button
          onClick={handleExport}
          disabled={moves.length === 0}
          className="btn btn-primary"
        >
          Export ({moves.length})
        </button>
        <button
          onClick={clearMoves}
          disabled={moves.length === 0}
          className="btn btn-danger"
        >
          Clear All
        </button>
      </div>

      {/* Keyboard Shortcuts Help */}
      <div className="moves-shortcuts">
        <strong>Shortcuts:</strong><br/>
        <kbd>Q</kbd> LH Start · <kbd>W</kbd> RH Start · <kbd>E</kbd> LH Fin · <kbd>R</kbd> RH Fin<br/>
        <kbd>Space</kbd> Submit · <kbd>C</kbd> Clear
      </div>
    </div>
  );
}

MovesPanel.propTypes = {
  moveParams: PropTypes.shape({
    currentMove: PropTypes.shape({
      lh_start: PropTypes.number,
      rh_start: PropTypes.number,
      lh_finish: PropTypes.array.isRequired,
      rh_finish: PropTypes.array.isRequired
    }).isRequired,
    resetCurrentMove: PropTypes.func.isRequired,
    moveCursorMode: PropTypes.string.isRequired,
    setMoveCursorMode: PropTypes.func.isRequired,
    isCurrentMoveValid: PropTypes.func.isRequired,
    addCurrentMoveToMoves: PropTypes.func.isRequired,
    moves: PropTypes.array.isRequired,
    removeMove: PropTypes.func.isRequired,
    clearMoves: PropTypes.func.isRequired,
    exportMoves: PropTypes.func.isRequired
  }).isRequired,
  holds: PropTypes.array.isRequired
};

export default MovesPanel;