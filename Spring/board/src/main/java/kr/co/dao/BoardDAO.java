package kr.co.dao;

import java.util.List;

import kr.co.vo.BoardVO;

public interface BoardDAO {
	// �Խñ� �ۼ�
	public void write(BoardVO boardVO) throws Exception;

	// �Խù� ��� ��ȸ
	public List<BoardVO> list() throws Exception;
}