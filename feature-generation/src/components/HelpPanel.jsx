function HelpPanel() {
  return (
    <div className="help-panel">
      <h3>Controls</h3>
      <ul>
        <li><kbd>Scroll</kbd> to zoom</li>
        <li><kbd>Drag</kbd> to pan (Pan mode)</li>
        <li><kbd>Click</kbd> to remove holds</li>
        <li><kbd>F</kbd> to fit image to window</li>
        <li><kbd>1-4</kbd> to switch modes</li>
      </ul>
      <h3 style={{ marginTop: '12px' }}>Adding Holds</h3>
      <ul>
        <li><strong>Click + Hold</strong> at hold position</li>
        <li><strong>Drag</strong> to set pull direction</li>
        <li><strong>Release</strong> to confirm</li>
        <li>Drag distance = useability (1-10)</li>
      </ul>
      <h3 style={{ marginTop: '12px' }}>Setting Climbs</h3>
      <h4>Climbs are set as a sequence of Positions</h4>
      <ul>
        <li><strong>Click</strong> on a hold to add that hold to the current position</li>
        <li><kbd>V</kbd> to switch the active limb</li>
        <li><kbd>C</kbd> to clear the position</li>
        <li><kbd>Spacebar</kbd> to add the position to the climbing sequence</li>
        <li><kbd>Enter</kbd> to save the climb and add it to the set climbs list</li>
        <li>Drag distance = useability (1-10)</li>
      </ul>
    </div>
  );
}

export default HelpPanel;
