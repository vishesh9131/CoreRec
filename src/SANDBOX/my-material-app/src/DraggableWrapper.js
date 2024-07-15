import React, { useRef } from 'react';
import Draggable from 'react-draggable';

const DraggableWrapper = ({ children, ...props }) => {
  const nodeRef = useRef(null);

  return (
    <Draggable nodeRef={nodeRef} {...props}>
      <div ref={nodeRef}>
        {children}
      </div>
    </Draggable>
  );
};

export default DraggableWrapper;