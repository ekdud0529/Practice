package kr.co.service;

import java.util.List;

import kr.co.vo.BoardVO;

public interface BoardService {
	// �Խñ� �ۼ�
	public void write(BoardVO boardVO) throws Exception;
	
	// �Խù� ��� ��ȸ
	public List<BoardVO> list() throws Exception;
}