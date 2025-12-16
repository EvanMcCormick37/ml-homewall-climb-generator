function HelpPanel({mode}) {
  return (
    <div className="help-panel">
      {(mode === 'hold' || mode === 'foot' || mode === 'remove') && (
        <>
        <h3 style={{ marginBottom: '12px' }}>Adding Holds</h3>
        <ul>
          <li><strong>Click + Hold</strong> at hold position</li>
          <li><strong>Drag</strong> to set pull direction</li>
          <li><strong>Release</strong> to confirm</li>
          <li>Drag distance = useability (1-10)</li>
        </ul>
        <h3 style={{ marginTop: '12px' }}>Removing Holds</h3>
          <ul>
            <li><strong>Click</strong> an existing hold to remove it</li>
          </ul>
        </>
      )}
      {mode === 'climb' && 
        <>
        <h3 style={{ marginBottom: '12px' }}>Setting Climbs</h3>
        <h4>Climbs are set as a sequence of Positions</h4>
        <ul>
          <li><strong>Click</strong> on a hold to add that hold to the current position</li>
          <li><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd><kbd>F</kbd> to switch the active limb <strong>[LH,RH,LF,RF]</strong></li>
          <li><kbd>V</kbd> to cycle the active limb</li>
          <li><kbd>X</kbd> to detach the previous limb from the wall</li>
          <li><kbd>C</kbd> to clear the position</li>
          <li><kbd>R</kbd> to roll-back the last position from climb sequence</li>
          <li><kbd>Spacebar</kbd> to add the position to the climbing sequence</li>
          {/* <li><kbd>Enter</kbd> to save the climb and add it to the set climbs list</li> */}
          <li>Drag distance = useability (1-10)</li>
        </ul>
        </>
      }
      <h3 style={{ marginTop: '12px' }}>Basic Controls</h3>
      <ul>
        <li><kbd>Scroll</kbd> to zoom</li>
        <li><kbd>Drag</kbd> to pan (Pan mode)</li>
        <li><kbd>W</kbd> to fit image to window</li>
        <li><kbd>1-6</kbd> to switch modes</li>
      </ul>
    </div>
  );
}

export default HelpPanel;
