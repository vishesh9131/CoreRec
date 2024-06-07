import React, { useRef } from 'react';
import { ResizableBox } from 'react-resizable';

const ResizableWrapper = ({ children, ...props }) => {
  const nodeRef = useRef(null);

  return (
    <ResizableBox {...props}>
      <div ref={nodeRef}>
        {children}
      </div>
    </ResizableBox>
  );
};

export default ResizableWrapper;