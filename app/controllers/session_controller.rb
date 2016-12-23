class SessionController < ApplicationController
  skip_before_action :verify_authenticity_token

  def index
  end

  def create
    if params[:authorisation_token] == Rails.application.secrets.authorisation_token
      reset_session
      session[:logged_in] = true
      redirect_to data_url
    else
      flash[:notice] = 'Invalid authorization token. Please try again.'
      redirect_to root_url
    end
  end

end
